from pandasai.llm.local_llm import LocalLLM
from pandasai import SmartDataframe
import streamlit as st
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
from dotenv import load_dotenv
from langchain_community.llms import Ollama
# import sqlite3
import os
import pickle

load_dotenv()
st.set_page_config(layout='wide')

with st.sidebar:
    
    select = option_menu('Machine Learning',
                        ['LLM Model (Local without fine tune)',
                         'LLM Model (By using API)',
                        'Heart Disease Prediction',
                        'Read Me',],
                        default_index=0)


if select == 'LLM Model (Local without fine tune)':
    def load_dataset(file):
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        elif file.name.endswith('.xlsx'):
            return pd.read_excel(file)
        else:
            raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")

    def main():
        st.title('LLM Model (Local without fine tune)')
        
        llm = Ollama(model="llama3.1:latest")

        uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=['csv', 'xlsx'])

        if uploaded_file is not None:
            try:
                df = load_dataset(uploaded_file)
                st.write("### Dataset Preview:")
                st.dataframe(df.head())

                smart_df = SmartDataframe(df, config={"llm": llm})

                user_query = st.text_input("Enter your question about the dataset:")
                generate_button = st.button("Generate")

                if generate_button:
                    if user_query:
                        try:
                            with st.spinner("Generating response..."):
                                response = smart_df.chat(user_query)
                            st.write("Response:", response)
                        except Exception as e:
                            st.error(f"Error querying the dataframe: {str(e)}")
                    else:
                        st.warning("Please enter a question before generating a response.")
            except Exception as e:
                st.error(f"Error processing the file: {str(e)}")
        else:
            st.info("Please upload a CSV or Excel file to proceed.")

    if __name__ == "__main__":
        main()


if select == 'LLM Model (By using API)':
    # st.title("Project Machine Learning by Vo Bao Long and Duy Truong")
    Gemini_API_KEY = os.environ['GEMINI_API_KEY']
    
    # File uploader for dataset
    uploaded_file = st.file_uploader("Upload your dataset as csv file", type=['csv'])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write(data.head(5))  
    
        df = SmartDataframe(data, config={"llm": model})
        prompt = st.text_area("Enter your question with csv file: ")
    
    if st.button("Generate:"):
        if prompt:
            with st.spinner("Waiting..."):
                st.write(df.chat(prompt))
# Load the pre-trained model for heart disease prediction



if (select == 'Heart Disease Prediction'):
    st.title('Heart Disease Prediction Model')
    loaded_model = pickle.load(open('model_heart_disease.pkl','rb'))


    # Function for making predictions
    def heart_disease_prediction(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
        input_data = (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
        input_data_to_model = np.asarray(input_data, dtype=np.float32)
        input_data_reshaped = input_data_to_model.reshape(1, -1)
        
        # Predict with the loaded model
        prediction = loaded_model.predict(input_data_reshaped)
        
        # Return prediction results
        if prediction[0] == 0:
            return 'Negative (No heart disease)'
        else:
            return 'Positive (Heart disease detected)'

    # Main function for Streamlit app
    def main():
        # User input fields for each feature
        age = st.text_input("Your age: ")
        sex = st.text_input("Male (1) / Female (0): ")
        cp = st.text_input("Chest pain type (0-3): ")
        trestbps = st.text_input("Resting blood pressure: ")
        chol = st.text_input("Serum cholesterol in mg/dl: ")
        fbs = st.text_input("Fasting blood sugar > 120 mg/dl (1 = true, 0 = false): ")
        restecg = st.text_input("Resting electrocardiographic results (0-2): ")
        thalach = st.text_input("Maximum heart rate achieved: ")
        exang = st.text_input("Exercise induced angina (1 = yes, 0 = no): ")
        oldpeak = st.text_input("ST depression induced by exercise relative to rest: ")
        slope = st.text_input("Slope of the peak exercise ST segment (0-2): ")
        ca = st.text_input("Number of major vessels (0-4) colored by fluoroscopy: ")
        thal = st.text_input("Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect): ")
        
        # Prediction result
        diagnosis = ''
        
        # Create a button for prediction
        if st.button('Test'):
            # Make sure to convert inputs to float where needed
            diagnosis = heart_disease_prediction(
                float(age), int(sex), int(cp), float(trestbps), float(chol), int(fbs), 
                int(restecg), float(thalach), int(exang), float(oldpeak), int(slope), 
                int(ca), int(thal)
            )
            
        st.success(diagnosis)


    if __name__ == '__main__':
        main()

if select == 'Read Me':
    st.title('This will have you understood about our project! ()')