import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

st.title("Streamlit ML classification Web App")

st.write("Which one is the best classifier?")

dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine"))

dataset_name = st.sidebar.selectbox("Select Classifier", ("KNN", "SVM", "Random Forest"))

# function to get the datasets
def get_dataset(dataset_name):
    # load datasets depending on user choice
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    
    # Split chosen data into data matrix and labeled target vector
    X = data.data
    y = data.target
    return X, y

X, y = get_dataset(dataset_name)
st.write("shape of dataset", X.shape)
st.write("number of classes", len(np.unique(y)))