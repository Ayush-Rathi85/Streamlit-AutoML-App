# importing required Libraries
import pandas as pd
import streamlit as st
import os

# Creating Menu bar
with st.sidebar:
    st.image('image.png')
    st.title("AutoML App")
    choice = st.radio("Menu", ['Upload Data', 'ML Model Training', 'Download Model'])
    st.info("This application builds machine learning model automatically using pycaret, pandas profiling and streamlit")

# Reading file if saved locally
if os.path.exists('file.csv'):
    df = pd.read_csv('file.csv')

# Uploading data in our app
if choice == 'Upload Data':
    st.title("Upload your data for modelling")
    file = st.file_uploader("Upload your dataset (it should be in csv format)")
    if file:
        df = pd.read_csv(file, index_col=False)
        df.to_csv('file.csv', index=False)
        st.dataframe(df)


# Training model according to users choice
if choice == 'ML Model Training':
    st.title("Model Training")
    target = st.selectbox("Select your target", df.columns) # Selecting Target class for model training
    Model_type = st.radio("Select Type of Model", ['Regression', 'Classification']) # choosing type of ML Problem

# Passing the data to the selected problem type
    if Model_type == 'Regression': # for regression problem
        from pycaret.regression import setup, compare_models, pull, save_model
        setup(df, target=target, silent=True)
        if st.button("Train Model"):
            setup_df = pull()
            st.info("These are the ML Model testing settings")
            st.dataframe(setup_df)
            best_model = compare_models()
            compare_df = pull()
            st.info("The best model for the data set: ")
            st.dataframe(compare_df)
            save_model(best_model, 'best_model')

    if Model_type == 'Classification': # for classification problem
        from pycaret.classification import setup, compare_models, pull, save_model
        setup(df, target=target, silent=True)
        if st.button("Train Model"):
            setup_df = pull()
            st.info("These are the ML Model testing settings")
            st.dataframe(setup_df)
            best_model = compare_models()
            compare_df = pull()
            st.info("The best model for the data set: ")
            st.dataframe(compare_df)
            save_model(best_model,'best_model')

# Providing Downloadable ML model
if choice == 'Download Model':
    with open("best_model.pkl", 'rb') as f:
        st.download_button('Download the model', f, "Auto_ML_Model.pkl")
