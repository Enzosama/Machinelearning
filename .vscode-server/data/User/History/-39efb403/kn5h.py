from langchain_community.vectorstores import Pinecone, FAISS
from langchain_community.llms import CTransformers
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import pinecone
import os
import sys

loader = CSVLoader(file_path="data/dataset.csv", encoding="utf-8", csv_args={'delimiter': ','})
data = loader.load()
# print(data)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
text_chunks = text_splitter.split_documents(data)
print(len(text_chunks))