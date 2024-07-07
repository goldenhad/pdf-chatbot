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

from utils import QAEntry
import os
import json

load_dotenv()

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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


embeddings = SpacyEmbeddings(model_name="en_core_web_sm")

# Twinstar Modified 2024.7.4
def vector_store(qas):
    text_chunks = [qa.question for qa in qas]
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    # Save the additional data along with the vector store
    with open("faiss_db_eng_metadata.json", "w") as f:
        json.dump([qa.to_dict() for qa in qas], f)

    vector_store.save_local("faiss_db_eng")


def get_conversational_chain(tools, ques, chat_history):
    api_key = os.environ["OPENAI_API_KEY"]
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0,
                     api_key=api_key)

    # Prepare the chat history messages for the prompt
    prompt_messages = [
        ("system",
         "You are a helpful assistant. Answer the question as detailed as possible from the provided context. If the answer is not in the provided context, just say, 'answer is not available in the context'. Don't provide the wrong answer."),
    ]

    for message in chat_history:
        if message['role'] == 'human':
            prompt_messages.append(("human", message['content']))
        elif message['role'] == 'assistant':
            prompt_messages.append(("assistant", message['content']))

    prompt_messages.append(("human", ques))

    prompt = ChatPromptTemplate.from_messages(
        prompt_messages + [("placeholder", "{agent_scratchpad}")]
    )

    tool = [tools]
    agent = create_tool_calling_agent(llm, tool, prompt)

    agent_executor = AgentExecutor(agent=agent, tools=tool, verbose=True)
    response = agent_executor.invoke({"input": ques, "chat_history": chat_history})
    return response['output']

# Twinstar Modified 2024.7.4
def user_input(user_question, chat_history, user_feedback=None):
    # Load the vector store
    new_db = FAISS.load_local("faiss_db_eng", embeddings, allow_dangerous_deserialization=True)

    # Load the additional data
    with open("faiss_db_eng_metadata.json", "r") as f:
        qas = [QAEntry.from_dict(data) for data in json.load(f)]

    retriever = new_db.as_retriever()
    retrieval_chain = create_retriever_tool(retriever, "pdf_extractor",
                                            "This tool is to give answer to queries from the pdf")
    response = get_conversational_chain(retrieval_chain, user_question, chat_history)

    # Get the top k responses
    # results = retriever.retrieve(user_question, k=10)

    # Important : Voting Algorithm
    # Adjust the results based on voting history
    # results_with_scores = []
    # for result in results:
    #     for qa in qas:
    #         if qa.question == result["text"]:
    #             score = result["score"]
    #             score += (qa.yes_votes - qa.no_votes)  # Adjust score based on votes
    #             results_with_scores.append((result, score))
    #             break

    # Sort by adjusted score
    # results_with_scores.sort(key=lambda x: x[1], reverse=True)
    # best_result = results_with_scores[0][0]

    # Find the corresponding QA entry and update voting history if needed
    # for qa in qas:
    #     if qa.question == best_result["text"]:
    #         if user_feedback == "yes":
    #             qa.yes_votes += 1
    #         elif user_feedback == "no":
    #             qa.no_votes += 1
    #         break
    # # Save the updated data
    # with open("faiss_db_eng_metadata.json", "w") as f:
    #     json.dump([qa.to_dict() for qa in qas], f)
    # response = {"question": best_result["text"], "answer": best_result["metadata"]["answer"]}

    return response


def main():
    st.set_page_config("Chat PDF")
    st.header("Interactive Chatbot with PDF")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if 'votes' not in st.session_state:
        st.session_state.votes = []

    with st.sidebar:
        pdf_doc = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button",
                                   accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = pdf_read(pdf_doc)
                text_chunks = get_chunks(raw_text)
                vector_store(text_chunks)
                st.success("Done")

    # Using a form to handle the 'Enter' hotkey
    with st.form(key='user_question_form', clear_on_submit=True):
        user_question = st.text_input("Ask a Question from the PDF Files", key="user_question_input")
        submit_button = st.form_submit_button(label='Send')

    if submit_button:
        if user_question:
            response = user_input(user_question, st.session_state.chat_history)
            st.session_state.chat_history.append(
                {"role": "human", "content": user_question})
            st.session_state.chat_history.append(
                {"role": "assistant", "content": response})
            st.session_state.votes.append({"upvotes": 0, "downvotes": 0})  # Ensure votes are appended for each response




    # if st.session_state.chat_history:
    #     for i, chat in enumerate(st.session_state.chat_history):
    #         print('-------------------------------->')
    #         print(i, chat)
    #         print(st.session_state)
    #
    #         if chat['role'] == 'human':
    #             st.write(f"**You:** {chat['content']}")
    #         elif chat['role'] == 'assistant':
    #             st.write(f"**ChatBot:** {chat['content']}")
    #
    #             col1, col2, col3 = st.columns([1, 1, 8])
    #             if col1.button("👍", key=f"up_{i}"):
    #                 print('upvoted',chat)
    #             #     st.session_state.votes[i]["upvotes"] += 1
    #             if col2.button("👎", key=f"down_{i}"):
    #                 print('downvoted', chat)
    #             #     st.session_state.votes[i]["downvotes"] += 1

            print(st.session_state)
                # col3.write(
                #     f"Upvotes: {st.session_state.votes[i]['upvotes']}, Downvotes: {st.session_state.votes[i]['downvotes']}")



if __name__ == "__main__":
    main()
