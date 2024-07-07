import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import AgentExecutor, create_tool_calling_agent

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

load_dotenv()

def pdf_read(pdf_doc):
    text = ""
    for pdf in pdf_doc:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks


embeddings = SpacyEmbeddings(model_name="fr_core_news_lg")


def vector_store(text_chunks):
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_db_eng")


def get_conversational_chain(tools, ques):
    api_key = os.environ["OPENAI_API_KEY"]
    # llm = ChatAnthropic(
    # model="claude-3-sonnet-20240229", temperature=0, api_key=os.getenv("ANTHROPIC_API_KEY"),verbose=True)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, api_key=api_key )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful assistant. Answer the question as detailed as possible from the provided 
                context, The context is extracted from my videos, make sure to provide all the details, if the answer is not in provided context just say, 
                "answer is not available in the context", don't provide the wrong answer""",
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    tool = [tools]
    agent = create_tool_calling_agent(llm, tool, prompt)

    agent_executor = AgentExecutor(agent=agent, tools=tool, verbose=True)
    response = agent_executor.invoke({"input": ques})

    st.session_state['response'] = response['output']
    return response

def user_input(user_question):
    new_db = FAISS.load_local("faiss_db_eng", embeddings, allow_dangerous_deserialization=True)
    retriever = new_db.as_retriever()
    retrieval_chain = create_retriever_tool(retriever, "pdf_extractor",
                                            "This tool is to give answer to queries from the pdf")
    response = get_conversational_chain(retrieval_chain, user_question)
    return response

def main():
    # if 'response' not in st.session_state:
    #     st.session_state['response'] = ""
    # if 'upvotes' not in st.session_state:
    #     st.session_state['upvotes'] = 0
    # if 'downvotes' not in st.session_state:
    #     st.session_state['downvotes'] = 0
    #
    # st.set_page_config("Chat PDF")
    # st.header("Chatbot v.1.0.2")
    # Using a form to handle the 'Enter' hotkey
    # with st.form(key='user_question_form', clear_on_submit=True):
    #     user_question = st.text_input("Ask a Question from the PDF Files", key="user_question_input")
    #     submit_button = st.form_submit_button(label='Send')

    bot_output = ""
    # if submit_button:
    #     if user_question:
    #         bot_output = user_input(user_question)
    #         st.write("Reply: ", st.session_state['response'])

    submit_button = st.button("Submit ")
    if submit_button:
        print("Send")
    # col1, col2, col3 = st.columns([1, 1, 8])
    # if col1.button("👍", key=f"up"):
    #     print("upvoted")
    #     # st.session_state['upvotes'] += 1
    #     # st.write("Upvotes: ", st.session_state['upvotes'])
    # if col2.button("👎", key=f"down"):
    #     # st.session_state['downvotes'] += 1
    #     print("downvoted")

        # st.write("Downvotes: ", st.session_state['downvotes'])
    # with st.sidebar:
    #     pdf_doc = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button",
    #                                accept_multiple_files=True)
    #     if st.button("Submit & Process"):
    #         with st.spinner("Processing..."):
    #             raw_text = pdf_read(pdf_doc)
    #             text_chunks = get_chunks(raw_text)
    #             vector_store(text_chunks)
    #             st.success("Done")


if __name__ == "__main__":

    main()
