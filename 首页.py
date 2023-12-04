import streamlit as st
import os

from langchain.memory import ConversationBufferMemory
from langchain_core.callbacks import StreamingStdOutCallbackHandler, BaseCallbackHandler

from utils import init_session_state
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.chains.llm import LLMChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT
from langchain.chains.question_answering import load_qa_chain


class MyCallbackHandler(BaseCallbackHandler):
    def __init__(self, res_placeholder):
        super(MyCallbackHandler, self).__init__()
        self.res_placeholder = res_placeholder
        self.res = ''


    def on_llm_new_token(self, token, **kwargs) -> None:
        # print every token on a new line
        self.res += token
        self.res_placeholder.markdown(self.res)


def main():
    # 初始化session_state
    init_session_state()

    st.title("Let's Chat!")
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "你好，我是一个聊天机器人，你可以向我问知识库中的内容"}
        ]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    chat_history = [("", "你好，我是一个聊天机器人，你可以向我问知识库中的内容")]

    if prompt := st.chat_input():
        if not st.session_state.llm_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()

        # 展示用户发送的信息
        with st.chat_message("user"):
            st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

        # 展示回答的信息
        with st.chat_message("assistant"):
            message_placeholder = st.empty()

        llm = OpenAI(temperature=0)
        # 关键在于要重写这个callback
        streaming_llm = OpenAI(streaming=True, callbacks=[MyCallbackHandler(message_placeholder)], temperature=0)

        question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
        doc_chain = load_qa_chain(streaming_llm, chain_type="stuff", prompt=QA_PROMPT)

        qa = ConversationalRetrievalChain(
            retriever=st.session_state.vector_db.as_retriever(), combine_docs_chain=doc_chain, question_generator=question_generator)

        result = qa({"question": prompt, "chat_history": chat_history})
        chat_history.append((prompt, result['answer']))
        st.session_state.messages.append({"role": "assistant", "content": result['answer']})


if __name__ == "__main__":
    main()
