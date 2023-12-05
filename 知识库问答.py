import streamlit as st
import os
from langchain.memory import ConversationBufferMemory
from utils import init_session_state, StreamingResponseCallbackHandler
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.chains.llm import LLMChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback



def main():
    # 初始化session_state
    init_session_state()

    st.title("Let's Chat!")
    if "knowledge_messages" not in st.session_state:
        st.session_state.knowledge_messages = [
            {"role": "assistant", "content": "你好，我是一个聊天机器人，你可以向我问知识库中的内容"}
        ]

    for msg in st.session_state.knowledge_messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    knowledge_history = [("", "你好，我是一个聊天机器人，你可以向我问知识库中的内容")]

    if prompt := st.chat_input():
        if not st.session_state.llm_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()

        # 展示用户发送的信息
        with st.chat_message("user"):
            st.markdown(prompt)
            st.session_state.knowledge_messages.append({"role": "user", "content": prompt})

        # 展示回答的信息
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
        llm = OpenAI(
            openai_api_key=st.session_state.llm_api_key,
            temperature=0, 
            model='gpt-3.5-turbo-instruct')
        
        # 关键在于要重写这个callback
        streaming_llm = OpenAI(
            openai_api_key=st.session_state.llm_api_key,
            streaming=True,
            callbacks=[StreamingResponseCallbackHandler(message_placeholder)],
            temperature=0,
            model='gpt-3.5-turbo-instruct')


        question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
        doc_chain = load_qa_chain(streaming_llm, chain_type="stuff", prompt=QA_PROMPT)

        qa = ConversationalRetrievalChain(
            retriever=st.session_state.vector_db.as_retriever(),
            combine_docs_chain=doc_chain,
            question_generator=question_generator)

        result = qa({"question": prompt, "chat_history": knowledge_history})
        knowledge_history.append((prompt, result['answer']))
        st.session_state.knowledge_messages.append({"role": "assistant", "content": result['answer']})


if __name__ == "__main__":
    main()
