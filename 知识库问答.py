import streamlit as st
from utils import init_session_state, StreamingResponseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain.chains.conversational_retrieval.prompts import (
    CONDENSE_QUESTION_PROMPT,
    QA_PROMPT,
)
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts.prompt import PromptTemplate


# 参考实现: https://python.langchain.com.cn/docs/use_cases/question_answering/how_to/chat_vector_db


def main():
    # 初始化session_state
    init_session_state()

    st.title("Let's Chat!")
    if "knowledge_messages" not in st.session_state:
        st.session_state.knowledge_messages = [("", "你好，我是一个聊天机器人，你可以向我问知识库中的内容")]

    for i in range(len(st.session_state.knowledge_messages)):
        if i == 0:
            st.chat_message("assistant").markdown(
                st.session_state.knowledge_messages[i][1]
            )
        else:
            st.chat_message("user").markdown(st.session_state.knowledge_messages[i][0])
            st.chat_message("assistant").markdown(
                st.session_state.knowledge_messages[i][1]
            )

    if prompt := st.chat_input():
        if not st.session_state.llm_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()

        # 展示用户发送的信息
        with st.chat_message("user"):
            st.markdown(prompt)

        # 展示回答的信息
        with st.chat_message("assistant"):
            message_placeholder = st.empty()

        # llm = ChatOpenAI(
        #     openai_api_key=st.session_state.llm_api_key,
        #     temperature=0,
        #     model="gpt-3.5-turbo",
        #     streaming=True,
        #     callbacks=[StreamingResponseCallbackHandler(message_placeholder)],
        # )
        # qa = ConversationalRetrievalChain.from_llm(
        #     llm,
        #     st.session_state.vector_db.as_retriever(),
        #     return_source_documents=True,
        #     return_generated_question=True,
        #     chain_type="stuff",
        # )
        # result = qa(
        #     {
        #         "question": "请回答下面的问题, 并且结果应该尽可能地详细和全面, 并且回答的格式应当正确且易读, 回答的语言应当和提问的语言相同:"
        #         + prompt,
        #         "chat_history": st.session_state.knowledge_messages[1:],
        #     }
        # )
        llm = ChatOpenAI(temperature=0)
        streaming_llm = ChatOpenAI(
            streaming=True,
            callbacks=[StreamingResponseCallbackHandler(message_placeholder)],
            temperature=0,
        )

        # 构建问题生成器的
        question_prompt_template = """根据下面的对话历史和问题, 提取为单独的新问题,新问题的语言与原问题相同.
        如果问题太短,需要将问题进行增强
        对话历史如下:{chat_history}
        提出的问题为: {question}
        生成的新问题为:"""
        question_prompt = PromptTemplate.from_template(question_prompt_template)
        question_generator = LLMChain(llm=llm, prompt=question_prompt)

        # 构建知识库检索链
        qa_prompt_template = """使用下面参考的内容来回答末尾的问题,如果你不知道问题的答案,就说不知道,不要编造答案.
        结果应该尽可能地详细和全面, 并且回答的格式应当正确且易读, 回答的语言应当和提问的语言相同.
        参考的内容为:{context}
        问题为: {question}
        回答为:"""
        qa_prompt = PromptTemplate(
            template=qa_prompt_template, input_variables=["context", "question"]
        )

        doc_chain = load_qa_chain(streaming_llm, chain_type="stuff", prompt=qa_prompt)

        qa = ConversationalRetrievalChain(
            retriever=st.session_state.vector_db.as_retriever(),
            combine_docs_chain=doc_chain,
            question_generator=question_generator,
            return_source_documents=True,
            return_generated_question=True,
        )
        result = qa(
            {
                "question": prompt,
                "chat_history": st.session_state.knowledge_messages[1:],
            }
        )
        print(result["source_documents"])
        print(result["answer"])
        print(result["generated_question"])
        st.session_state.knowledge_messages.append((prompt, result["answer"]))


if __name__ == "__main__":
    main()
