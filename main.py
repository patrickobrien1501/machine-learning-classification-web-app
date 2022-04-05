from argon2 import Parameters
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

import numpy as np
from sklearn import datasets

st.title("Streamlit ML classification Web App")

st.write("Which one is the best classifier?")

dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine"))

classifier_name = st.sidebar.selectbox("Select Classifier", ("KNN", "SVM", "Random Forest"))

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

def add_parameter_ui(clf_name):
    params = {}
    if clf_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K
    elif clf_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C
    else:
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
    return params

params = add_parameter_ui(classifier_name)

def get_classifier(clf_name, params):
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])

    elif clf_name == "SVM":
        clf = SVC(C=params["C"])

    else:
        clf = RandomForestClassifier(n_estimators=params["n_estimators"],
                                    max_depth=params["max_depth"], random_state=1234)
    return clf

clf = get_classifier(classifier_name, params)

# Classification
# split the test-train data with custom split ratio
# test_size = st.slider("test-train-ratio:", 0.1, 0.3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1234)

# The API for the sklearn algorithms is the same for all classifiers
clf.fit(X_train, y_train)
y_prediction = clf.predict(X_test)

# print out the accuracy scores
accuracy = accuracy_score(y_test, y_prediction)
st.write(f"classifier = {classifier_name}")
st.write(f"accuracy = {accuracy}")

# Plot classification results
# Apply PCA to have 2D representation of our results
# specify that we want to keep 2 dimensions 
pca = PCA(2)
X_projected = pca.fit_transform(X) # unsupervised, so does not need y here

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

# create figure object
fig = plt.figure(figsize=(4,3)) 
# color based on labels, alpha value for transparency
plt.scatter(x1, x2, c=y, alpha=0.8, cmap="viridis") 
plt.xlabel("Principle Component 1")
plt.ylabel("Principle Component 2")
plt.colorbar()

# plt.show()   # this is what we would usually need to plot
st.pyplot(fig)

# TODO
# - add more parameters (sklearn)
# - add other classifiers
# - add feature scaling
