

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score
import pandas as pd, numpy as np
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt


st.title("Bank Fraud Data Classification Web App")
st.set_option('deprecation.showPyplotGlobalUse', False)
st.sidebar.title("Binary Classification Web App")
st.markdown("Fraud Data yes or no?")
st.sidebar.markdown("Fraud Data yes or no?")


@st.cache(persist=True)

def load_data():

    fraud_detection = pd.read_csv('fraud_detection.csv')

    fraud_detection.gender = fraud_detection.gender.astype('category').cat.codes
    fraud_detection.category = fraud_detection.category.astype('category').cat.codes
    fraud_detection.customer = fraud_detection.customer.astype('category').cat.codes
    fraud_detection.merchant = fraud_detection.merchant.astype('category').cat.codes

    fraud_detection.age = fraud_detection.age.str.replace('[\'\,]', '', regex=True)
    fraud_detection.zipcodeOri = fraud_detection.zipcodeOri.str.replace('[\'\,]', '', regex=True)
    fraud_detection.zipMerchant = fraud_detection.zipMerchant.str.replace('[\'\,]', '', regex=True)
    fraud_detection["age"] = pd.to_numeric(fraud_detection["age"], errors='ignore')
    fraud_detection["zipcodeOri"] = pd.to_numeric(fraud_detection["zipcodeOri"], errors='coerce')
    fraud_detection["zipMerchant"] = pd.to_numeric(fraud_detection["zipMerchant"], errors='coerce')
    fraud_detection = fraud_detection[~(fraud_detection.age == '0')]
    fraud_detection = fraud_detection[~(fraud_detection.age == 'U')]
    return fraud_detection


@st.cache(persist=True)

def split(df):
    y = df.fraud
    x = df.drop(columns=['fraud'])
    x.drop(columns="zipcodeOri", inplace=True)
    x.drop(columns="zipMerchant", inplace=True)

    scaler=RobustScaler()
    x=pd.DataFrame(scaler.fit_transform(x),columns=x.columns)
    x.head()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)
    return x_train, x_test, y_train, y_test

def plot_metrics(model, metrics_list):
    if 'Confusion Matrix' in metrics_list:
        st.subheader("Confusion Matrix")
        plot_confusion_matrix(model, x_test, y_test, display_labels=class_names)
        st.pyplot()
    
    if 'ROC Curve' in metrics_list:
        st.subheader("ROC Curve")
        plot_roc_curve(model, x_test, y_test)
        st.pyplot()
    
    if 'Precision-Recall Curve' in metrics_list:
        st.subheader("Precision-Recall Curve")
        plot_precision_recall_curve(model, x_test, y_test)
        st.pyplot()


def choose_classifier(classifier):

    if classifier == 'Gradient Boosting':
        st.sidebar.subheader("Model Hyperparameters")

        n_estimators = st.sidebar.number_input("The number of trees in the forest : ", 20, 100, step=10, key='n_estimators')
        max_depth = st.sidebar.number_input("The maximum depth of the tree : ", 1, 15, step=1, key='max_depth')
        bootstrap = st.sidebar.radio("Bootstrap samples when building trees : ", ('True', 'False'), key='bootstrap')

        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key = 'classify'):
            st.subheader("Gradient Boosting Results")
            model = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=0)
            model.fit(x_train,y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            pred = pd.DataFrame(y_pred)

            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test,y_pred, labels = class_names).round(2))
            st.write("Recall: ", recall_score(y_test,y_pred, labels = class_names).round(2))

            plot_metrics(model, metrics)

    if classifier == 'Logistic Regression':

        st.sidebar.subheader("Model Hyperparameters")

        C = st.sidebar.number_input("C (Regularisation parameter)", 0.01, 10.0, step = 0.01, key = "C_LR")
        max_iter = st.sidebar.slider("Maximum Number of iterations", 50, 200, key = 'max_iter')

        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key = 'classify'):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C, max_iter = max_iter)
            model.fit(x_train,y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)

            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test,y_pred, labels = class_names).round(2))
            st.write("Recall: ", recall_score(y_test,y_pred, labels = class_names).round(2))


            plot_metrics(model, metrics)

    if classifier == 'Random Forest':

        st.sidebar.subheader("Model Hyperparameters : ")
        n_estimators = st.sidebar.number_input("The number of trees in the forest : ", 100, 200, step=10, key='n_estimators')
        max_depth = st.sidebar.number_input("The maximum depth of the tree : ", 1, 15, step=1, key='max_depth')
        bootstrap = st.sidebar.radio("Bootstrap samples when building trees : ", ('True', 'False'), key='bootstrap')
        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)

            
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            

            plot_metrics(model, metrics)


if st.sidebar.checkbox("Show raw data", False):
    st.subheader("Fraud Detection")
    st.write(pd.read_csv('fraud_detection.csv'))
    
df = load_data()
x_train, x_test, y_train, y_test = split(df)
class_names = ['1', '0']
st.sidebar.subheader("Choose Classifier")
classifier = st.sidebar.selectbox("Classifier", ("Gradient Boosting", "Logistic Regression", "Random Forest"))

choose_classifier(classifier)
