from langchain_community.vectorstores import Pinecone, FAISS
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import pinecone
import os
import sys
from pandasai.llm.local_llm import LocalLLM  # Using LocalLLM for embedding

# Initialize Local LLM (llama3)
model = LocalLLM(
    api_base="http://localhost:11434/v1",  
    model="llama3"
)

# Load the CSV file
loader = CSVLoader(file_path="data/dataset.csv", encoding="utf-8", csv_args={'delimiter': ','})
data = loader.load()

# Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
text_chunks = text_splitter.split_documents(data)
print(f"Total chunks: {len(text_chunks)}")

# Function to convert text into embeddings using Local LLM
def get_embeddings_from_llm(texts):
    embeddings = []
    for text in texts:
        # Assuming LocalLLM has a method like generate_embeddings
        response = model.generate_embeddings(text)  # Use the method for generating embeddings
        embeddings.append(response)  # Assuming the response directly returns the embeddings
    return embeddings

# Converting text chunks into embeddings using Local LLM
text_embeddings = get_embeddings_from_llm([chunk.page_content for chunk in text_chunks])

# Store embeddings into FAISS Knowledge Base
docsearch = FAISS.from_embeddings(text_chunks, text_embeddings)
docsearch.save_local("DB_FAISS_PATH")  # Save FAISS index locally

# Sample query and search using FAISS
query = "How many 1 in target column"
docs = docsearch.similarity_search(query, k=3)
print("Result", docs)


print(dir(model))
