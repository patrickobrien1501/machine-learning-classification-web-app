from imports import *
from functions import get_dataset, add_parameter_ui, get_classifier

st.title("Streamlit ML classification Web App")

st.write("Which one is the best classifier?")

dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine"))
classifier_name = st.sidebar.selectbox("Select Classifier", ("KNN", "SVM", "Random Forest"))

if __name__ == "__main__":

    # Load dataset
    X, y = get_dataset(dataset_name)
    st.write("shape of dataset", X.shape)
    st.write("number of classes", len(np.unique(y)))

    # Pass classifier name to function and return parameter dictionary
    params = add_parameter_ui(classifier_name)

    # Return classifier object instance
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
