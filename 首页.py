import streamlit as st
import os
from utils import init_session_state


def main():
    # 初始化session_state
    init_session_state()

    st.title('本地知识库问答系统')
    # st.write(st.session_state)


if __name__ == "__main__":
    main()
