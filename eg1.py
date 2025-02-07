
import seaborn as sns
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle
from sklearn.svm import SVC
import gradio as gr

iris_data= sns.load_dataset("iris")

i=["sepal_length","sepal_width","petal_length","petal_width"]

X =iris_data[i]
y = iris_data["species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr_clf=LogisticRegression(random_state=42)
dr_clf=DecisionTreeClassifier()
svc_clf=SVC(probability=True)

voting_clf = VotingClassifier(estimators=[
    ('LogisticRegression', lr_clf),
    ('DecisionTree', dr_clf),
    ("svc",svc_clf)
], voting='hard')

voting_clf.fit(X_train,y_train)


with open("iris_model.pkl","wb") as file:
    pickle.dump(voting_clf,file)


def predict_iris(sepal_length,sepal_width,petal_length,petal_width):
    
    with open("iris_model.pkl","rb") as file:
        load_model=pickle.load(file)

    input_data=[[sepal_length,sepal_width,petal_length,petal_width]]
    prediction=load_model.predict(input_data)[0]

    return prediction

iface=gr.Interface(
    fn=predict_iris, 
    inputs=[gr.Number(label="sepal length (cm)"),
    gr.Number(label="sepal width (cm)"),
    gr.Number(label="petal length (cm)"),
    gr.Number(label="petal width (cm)")],

outputs="text",

title="Iris flower classifier",
description="Enter the flower's measurements to predict the class.")

iface.launch()