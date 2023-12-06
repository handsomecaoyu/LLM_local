import json
import os
import streamlit as st
from configs.config import USER_CONFIG_PATH, VECTOR_DB_PATH, INDEX_DB_URL
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import SQLRecordManager
from langchain_core.callbacks import BaseCallbackHandler


def save_data(data, file_path):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# 从 JSON 文件加载数据
def load_data(data_path):
    with open(data_path, "r") as f:
        data = json.load(f)
    return data


def find_all_files(path, target_file_types):
    res = []
    if os.path.exists(path):
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.split(".")[-1] in target_file_types:
                    res.append(os.path.join(root, file))
    return res


def init_session_state(state_path=USER_CONFIG_PATH):
    # 用户配置的api_key，知识库路径等

    user_config = load_data(state_path)
    for k, v in user_config.items():
        if k not in st.session_state:
            st.session_state[k] = v
        else:
            # 你可能觉得这行代码很奇怪，似乎做了一个完全没有用的操作
            # 但是如果没有这行代码，当在首页完成对话的时候，再跳转到配置页，那么会发现配置的内容不存在了，很奇怪的bug。
            # 这么重新赋值一下就能解决，怀疑是
            st.session_state[k] = st.session_state[k]

    # 配置向量数据库
    if "embedding_api_key" in st.session_state:
        embedding = OpenAIEmbeddings(openai_api_key=st.session_state.embedding_api_key)
        if "vector_db" not in st.session_state:
            st.session_state.vector_db = Chroma(
                persist_directory=VECTOR_DB_PATH, embedding_function=embedding
            )
    # 配置index数据库
    if "record_manager" not in st.session_state:
        st.session_state.record_manager = SQLRecordManager(
            "chroma_index", db_url=INDEX_DB_URL
        )
        st.session_state.record_manager.create_schema()


class StreamingResponseCallbackHandler(BaseCallbackHandler):
    def __init__(self, res_placeholder):
        super(StreamingResponseCallbackHandler, self).__init__()
        self.res_placeholder = res_placeholder
        self.res = ""

    def on_llm_new_token(self, token, **kwargs) -> None:
        # print every token on a new line
        self.res += token
        self.res_placeholder.markdown(self.res)
