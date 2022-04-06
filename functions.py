from imports import *

# function definitions
def get_dataset(dataset_name):
    '''
    Function to load datasets depending on user choice
    Input: dataset_name obtained from sidebar dropdown menu
    Outputs: Training data X and labeled output vector y

    '''
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

def add_parameter_ui(clf_name):
    '''
    Function to add classifier parameter(s) to a dictionary
    Input: Classifier name from sidebar drop down menu
    Output: Dictionary with classifier specific parameters

    '''
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

def get_classifier(clf_name, params):
    '''
    Applies parameters to respective instantiated classifier object
    Input: Classifier name and parameters dictionary
    Output: Classifier object instance

    '''
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])

    elif clf_name == "SVM":
        clf = SVC(C=params["C"])

    else:
        clf = RandomForestClassifier(n_estimators=params["n_estimators"],
                                    max_depth=params["max_depth"], random_state=1234)
    return clf