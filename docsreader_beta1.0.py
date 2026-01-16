import os
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from google import genai
from google.genai import types
import numpy as np


os.environ["GOOGLE_API_KEY"] = ""

model = init_chat_model("google_genai:gemini-2.5-flash-lite")

from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
from langchain_chroma import Chroma

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)

#========================File chunking part======================

file_path = "./Options_futures_derivatives.pdf"
file_path2="./budget_speech.pdf"
file_path3="./winwin.pdf"
loader = PyPDFLoader(file_path2)

docs = loader.load()
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
all_splits = text_splitter.split_documents(docs)

print(f"Split blog post into {len(all_splits)} sub-documents.")


#==storage in vectorDB=============

document_ids = vector_store.add_documents(documents=all_splits)

print(document_ids[:3])

from langchain.tools import tool

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


from langchain.agents import create_agent


tools = [retrieve_context]
# If desired, specify custom instructions
prompt = (
    "You have access to a tool that retrieves context from a blog post. "
    "Use the tool to help answer user queries."
)
agent = create_agent(model, tools, system_prompt=prompt)


'''

========================Query processing and IO============== 

'''
flag=True
while(True):
	x=input('1)To continue \n 2) to exit')
	if(x==2):
		break
	else:
		query = input('ASk your question: ')
		for event in agent.stream(
			{"messages": [{"role": "user", "content": query}]},
			stream_mode="values",
    
		):
			event["messages"][-1].pretty_print()


    		
		
		


