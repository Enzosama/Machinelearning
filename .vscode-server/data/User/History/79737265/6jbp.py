###
from Langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from Langchain.vectorstores import Pinecone
from Langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import pinecone
import os
import sys
###
from pandasai.llm.local_llm import LocalLLM
from pandasai import SmartDataframe
import streamlit as st
import pandas as pd
import pickle
import numpy as np
from streamlit_option_menu import option_menu
from dotenv import load_dotenv
from pandasai.llm.openai import OpenAI
from pandasai import PandasAI
####

loader = CSVLoader (file_path="data/dataset.csv", encoding="utf-8", csv_args={'delimiter': ','})
data = loader.load()
print(data)
# Split the text into Chunks
text_splitter = RecursiveCharacterTextSplitter (chunk_size=500, chunk_overlap=20)
text_chunks = text_splitter.split_documents (data)
print(len(text_chunks))

# load_dotenv()



# st.set_page_config(layout='wide')

# Commented out model loading for pandasai integration
# model = LocalLLM(
#     api_base="http://localhost:11434/v1",  
#     model="llama3" 
# )

# st.title("Project Machine Learning by Vo Bao Long and Duy Truong")

# File uploader for dataset
# uploaded_file = st.file_uploader("Upload your dataset as csv file", type=['csv'])

# if uploaded_file is not None:
#     data = pd.read_csv(uploaded_file)
#     st.write(data.head(5))  
    
#     df = SmartDataframe(data, config={"llm": model})
#     prompt = st.text_area("Enter your question with csv file: ")
    
#     if st.button("Generate:"):
#         if prompt:
#             with st.spinner("Waiting..."):
#                 st.write(df.chat(prompt))

# Load the pre-trained model for heart disease prediction



# loaded_model = pickle.load(open('model_heart_disease.pkl','rb'))

# # Function for making predictions
# def heart_disease_prediction(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
#     input_data = (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
#     input_data_to_model = np.asarray(input_data, dtype=np.float32)
#     input_data_reshaped = input_data_to_model.reshape(1, -1)
    
#     # Predict with the loaded model
#     prediction = loaded_model.predict(input_data_reshaped)
    
#     # Return prediction results
#     if prediction[0] == 0:
#         return 'Negative (No heart disease)'
#     else:
#         return 'Positive (Heart disease detected)'

# # Main function for Streamlit app
# def main():
#     st.title('Heart Disease Prediction Model')
    
#     # User input fields for each feature
#     age = st.text_input("Your age: ")
#     sex = st.text_input("Male (1) / Female (0): ")
#     cp = st.text_input("Chest pain type (0-3): ")
#     trestbps = st.text_input("Resting blood pressure: ")
#     chol = st.text_input("Serum cholesterol in mg/dl: ")
#     fbs = st.text_input("Fasting blood sugar > 120 mg/dl (1 = true, 0 = false): ")
#     restecg = st.text_input("Resting electrocardiographic results (0-2): ")
#     thalach = st.text_input("Maximum heart rate achieved: ")
#     exang = st.text_input("Exercise induced angina (1 = yes, 0 = no): ")
#     oldpeak = st.text_input("ST depression induced by exercise relative to rest: ")
#     slope = st.text_input("Slope of the peak exercise ST segment (0-2): ")
#     ca = st.text_input("Number of major vessels (0-4) colored by fluoroscopy: ")
#     thal = st.text_input("Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect): ")
    
#     # Prediction result
#     diagnosis = ''
    
#     # Create a button for prediction
#     if st.button('Test'):
#         # Make sure to convert inputs to float where needed
#         diagnosis = heart_disease_prediction(
#             float(age), int(sex), int(cp), float(trestbps), float(chol), int(fbs), 
#             int(restecg), float(thalach), int(exang), float(oldpeak), int(slope), 
#             int(ca), int(thal)
#         )
        
#     st.success(diagnosis)

# if __name__ == '__main__':
#     main()
