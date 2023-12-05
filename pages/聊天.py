import streamlit as st
from utils import init_session_state, StreamingResponseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)



def main():
    init_session_state()
    # 创建一个聊天的界面，
    st.title("Let's Chat!")
    # 选择使用gpt4.0还是gpt3.5
    st.selectbox('选择用于对话的模型', ['gpt-3.5-turbo', 'gpt-4-1106-preview'], key='chat_model')
    if "llm_api_key" not in st.session_state:
        st.info("请在配置页添加你的OpenAI API key")

    st.session_state.chat_model_instance = ChatOpenAI(
        openai_api_key=st.session_state.llm_api_key,
        model=st.session_state.chat_model
    )
    print(st.session_state.chat_model_instance)
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            AIMessage(content='你好，我是一个聊天机器人')
        ]

    # 展示聊天的信息
    for msg in st.session_state.chat_messages:
        st.chat_message('assistant' if isinstance(msg, AIMessage) else 'user').markdown(msg.content)


    # 用户输入信息
    if prompt := st.chat_input():
        if not st.session_state.llm_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()

        # 展示用户发送的信息
        with st.chat_message("user"):
            st.markdown(prompt)
            st.session_state.chat_messages.append(HumanMessage(content=prompt))

        # 展示回答的信息，先展示一个空的信息
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
        
        # 通过llm进行对话
        result = ''
        for chunk in st.session_state.chat_model_instance.stream(st.session_state.chat_messages):
            result += chunk.content
            message_placeholder.markdown(result)

        st.session_state.chat_messages.append(AIMessage(content=result))


if __name__ == '__main__':
    main()