from langchain_community.vectorstores import Pinecone, FAISS
from langchain_community.llms import CTransformers
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import pinecone
import os
import sys

from pandasai.llm.local_llm import LocalLLM
from pandasai import SmartDataframe
import streamlit as st
import pandas as pd
import pickle
import numpy as np
from streamlit_option_menu import option_menu
from dotenv import load_dotenv
from pandasai.llm.openai import OpenAI
# from pandasai import PandasAI

# Loading and splitting the dataset
loader = CSVLoader(file_path="data/dataset.csv", encoding="utf-8", csv_args={'delimiter': ','})
data = loader.load()
print(data)

# Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
text_chunks = text_splitter.split_documents(data)
print(len(text_chunks))