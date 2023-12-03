import streamlit as st
import os
from utils import init_session_state, save_data, find_all_files
from configs.config import USER_CONFIG_PATH


def main():
    init_session_state()
    st.title('配置')

    st.divider()

    st.header('Embedding模型')
    st.text_input(label='embedding_api_key',
                  placeholder='请输入用于embedding的Api Key，目前只能OpenAI的API',
                  label_visibility='hidden',
                  key='embedding_api_key')

    st.header('LLM模型')
    st.text_input(label='llm_api_key',
                  placeholder='请输入LLM的Api Key，目前只能OpenAI的API',
                  label_visibility='hidden',
                  key='llm_api_key')

    st.divider()

    st.header('本地知识库路径')
    st.text_area(label='knowledge_paths',
                 placeholder='请输入文件夹路径，每个路径一行，用回车进行分割',
                 label_visibility='hidden',
                 key='knowledge_paths')

    st.header('知识库中的文件类型')
    st.multiselect(
        '知识库文件类型',
        placeholder='请从下拉菜单中选择文件类型',
        key="file_types",
        label_visibility='hidden',
        options=['txt', 'md', 'pdf']
    )

    if st.button('构建知识库'):
        if len(st.session_state.knowledge_paths) == 0:
            st.warning('请先输入知识库路径')
        else:
            # 获得目录下的文件
            knowledge_dirs = st.session_state.knowledge_paths.split('\n')
            target_file_types = set(st.session_state.file_types)
            file_paths = []
            for knowledge_dir in knowledge_dirs:
                file_paths.extend(find_all_files(knowledge_dir, target_file_types))
            st.write(len(file_paths))

    save_data(st.session_state.to_dict(), USER_CONFIG_PATH)


if __name__ == "__main__":
    main()
