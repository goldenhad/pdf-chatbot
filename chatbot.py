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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks


# embeddings = SpacyEmbeddings(model_name="fr_core_news_lg")
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
#

def vector_store(text_chunks):
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_db")


def get_conversational_chain(tools, ques):
    openai_api_key = os.environ["OPENAI_API_KEY"]
    # llm = ChatAnthropic(
    # model="claude-3-sonnet-20240229", temperature=0, api_key=os.getenv("ANTHROPIC_API_KEY"),verbose=True)
    llm = ChatOpenAI(model_name="gpt-4", temperature=0, api_key=openai_api_key )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a Trading Instructor assistant. The context is about Trading. Answer the question as detailed as possible from the provided 
                context, The context was extracted from my video by human, make sure to provide all the detail answers to the question in the context, 
                if the answer is not in provided context, please provide the most relevant answer in the context with some statement, but in any cases don't provide the wrong answer""",
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
    print(response)
    st.write("Reply: ", response['output'])
    # col1, col2, col3 = st.columns([1, 1, 8])
    # if col1.button("👍", key=f"up"):
    #     print('upvoted',response)
    # #     st.session_state.votes[i]["upvotes"] += 1
    # if col2.button("👎", key=f"down"):
    #     print('downvoted', response)


def user_input(user_question):
    new_db = FAISS.load_local("faiss_db", embeddings, allow_dangerous_deserialization=True)
    retriever = new_db.as_retriever()
    retrieval_chain = create_retriever_tool(retriever, "pdf_extractor",
                                            "This tool is to give answer to queries from the pdf")
    get_conversational_chain(retrieval_chain, user_question)


def main():
    st.set_page_config("Chat PDF")
    st.header("Chatbot v.1.0.2")
    user_question = st.text_input("Ask a Question from the Media Transcript Files")
    if user_question:
        user_input(user_question)
    with st.sidebar:
        pdf_doc = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button",
                                   accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = pdf_read(pdf_doc)
                text_chunks = get_chunks(raw_text)
                vector_store(text_chunks)
                # vector_store(raw_text)
                st.success("Done")


if __name__ == "__main__":
    main()
