import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
import faiss
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_core.runnables import  RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import pdfplumber


# Streamlit app
st.title("RAG-based Chatbot with PDF Support")

# User input
uploaded_file = st.file_uploader("Upload PDF", type="pdf")
question = st.text_input("Enter your question:")


def extract_text_from_pdf(uploaded_file):
    with pdfplumber.open(uploaded_file) as pdf:
        pages = [page.extract_text() for page in pdf.pages]
    return pages


def split_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
    split_text = text_splitter.create_documents(text)
    return split_text

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def faiss(doc):
    docsearch=FAISS.from_documents(doc, AzureOpenAIEmbeddings(api_key = "4699cbb6884a438095ba926bc0a8e12d",azure_endpoint =  "https://sandbox-ey.openai.azure.com/",api_version = "2023-03-15-preview"))
    return docsearch
    

def retriever_vec(vector):
    retriever = vectorstore.as_retriever()
    return retriever

def ret_prompt(temp):
    template = """Use the following pieces of context to answer the question at the end. 
                    If you don't know the answer, just say that you don't know, don't try to make up an answer.
                    You are a Enviornment Social Governance specialist for the company for which the report is given to you.
                    Give answers as per a business perspective.
                    Don't give any assumptions.
                    Use 6 sentences maximum and keep the answer as concise as possible. 
                    {context}
                    Question: {question}
                    Helpful Answer:"""

    prompt =  PromptTemplate.from_template(template)
    return prompt

def azure_llm():
    llm2 = AzureChatOpenAI(deployment_name="gpt-35-turbo",api_key = "4699cbb6884a438095ba926bc0a8e12d",azure_endpoint =  "https://sandbox-ey.openai.azure.com/",api_version = "2023-03-15-preview")
    return llm2




# Generate response with RAG model

# Main logic
if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)
    text=split_documents(text)
    vectorstore = faiss(text)
    retriever=retriever_vec(vectorstore)
    prompt=ret_prompt(retriever)
    llm = AzureChatOpenAI(deployment_name="gpt-35-turbo",api_key = "4699cbb6884a438095ba926bc0a8e12d",azure_endpoint =  "https://sandbox-ey.openai.azure.com/",api_version = "2023-03-15-preview")

    

    rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
    )
    
    if question:
        st.write("User Question:", question)
        answer = rag_chain.invoke(question)
        st.write("Bot Answer:", answer)
