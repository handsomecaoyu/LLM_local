import streamlit as st
import os
from utils import init_session_state, save_data, find_all_files
from configs.config import USER_CONFIG_PATH
from langchain.document_loaders import (
    UnstructuredMarkdownLoader,
    PyMuPDFLoader,
    UnstructuredFileLoader,
)
from langchain.indexes import index
from langchain.text_splitter import RecursiveCharacterTextSplitter


def batch_save_to_vector_db(file_paths):
    docs = []
    loader_dict = {
        "md": UnstructuredMarkdownLoader,
        "pdf": PyMuPDFLoader,
        "txt": UnstructuredFileLoader,
    }
    # 加载文件
    error_files = []
    for file_path in file_paths:
        try:
            loader = loader_dict[file_path.split(".")[-1]](file_path)
            docs.extend(loader.load())
        except Exception as e:
            st.write(e)
            error_files.append(file_path)

    # 切分文档
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
    split_docs = text_splitter.split_documents(docs)

    result = index(
        split_docs,
        st.session_state.record_manager,
        st.session_state.vector_db,
        cleanup="incremental",
        source_id_key="source",
    )
    return result, error_files


def save_to_vector_db(file_paths, batch_size=20):
    process_bar = st.progress(0, text="正在处理第0个文件")
    if len(file_paths) > batch_size:
        result = {"num_added": 0, "num_updated": 0, "num_skipped": 0, "num_deleted": 0}
        error_files = []
        for i in range(0, len(file_paths), batch_size):
            # streamlit进度条显示
            process_bar.progress(i / len(file_paths), text="正在处理第{}个文件".format(i))
            tmp_res, tmp_err = batch_save_to_vector_db(file_paths[i : i + batch_size])
            result["num_added"] += tmp_res["num_added"]
            result["num_updated"] += tmp_res["num_updated"]
            result["num_skipped"] += tmp_res["num_skipped"]
            result["num_deleted"] += tmp_res["num_deleted"]
            error_files.extend(tmp_err)
    else:
        result, error_files = batch_save_to_vector_db(file_paths)
    process_bar.progress(1.0, text="处理完成")
    return result, error_files


def main():
    init_session_state()
    st.title("配置")

    st.divider()

    st.header("Embedding模型")
    st.text_input(
        label="embedding_api_key",
        placeholder="请输入用于embedding的Api Key，目前只能OpenAI的API",
        label_visibility="hidden",
        key="embedding_api_key",
    )

    st.header("LLM模型")
    st.text_input(
        label="llm_api_key",
        placeholder="请输入LLM的Api Key，目前只能OpenAI的API",
        label_visibility="hidden",
        key="llm_api_key",
    )

    st.divider()

    st.header("本地知识库路径")
    st.text_area(
        label="knowledge_paths",
        placeholder="请输入文件夹路径，每个路径一行，用回车进行分割",
        label_visibility="hidden",
        key="knowledge_paths",
    )

    st.header("知识库中的文件类型")
    st.multiselect(
        "知识库文件类型",
        placeholder="请从下拉菜单中选择文件类型",
        key="file_types",
        label_visibility="hidden",
        options=["txt", "md", "pdf"],
    )

    if st.button("构建知识库"):
        if len(st.session_state.knowledge_paths) == 0:
            st.warning("请先输入知识库路径")
        else:
            # 获得目录下的文件
            knowledge_dirs = st.session_state.knowledge_paths.split("\n")
            target_file_types = set(st.session_state.file_types)
            file_paths = []
            for knowledge_dir in knowledge_dirs:
                file_paths.extend(find_all_files(knowledge_dir, target_file_types))
            # 将目录下的文件存入向量数据库
            if "vector_db" in st.session_state:
                st.write("一共有{}个文件需要处理".format(len(file_paths)))
                save_db_res, save_db_err = save_to_vector_db(file_paths)
                st.write(
                    "增加了{}个chunks, 更新了{}个chunks, 跳过了{}个chunks, 删除了{}个chunks".format(
                        save_db_res["num_added"],
                        save_db_res["num_updated"],
                        save_db_res["num_skipped"],
                        save_db_res["num_deleted"],
                    )
                )
                if len(save_db_err) > 0:
                    st.write("处理失败的文件：")
                    st.write(save_db_err)

            else:
                st.write("没有向量数据库")

    user_config = {
        "embedding_api_key": st.session_state["embedding_api_key"],
        "llm_api_key": st.session_state["llm_api_key"],
        "knowledge_paths": st.session_state["knowledge_paths"],
        "file_types": st.session_state["file_types"],
    }
    save_data(user_config, USER_CONFIG_PATH)


if __name__ == "__main__":
    main()
