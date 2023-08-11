from flask import Flask,render_template,request
from flask import Flask, send_from_directory


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


result = ""

heart_data = pd.read_csv('heart_disease_data.csv')




print(heart_data.head())
print(heart_data.tail())
print(heart_data.shape)
print(heart_data.info())


head_data = heart_data.head()
tail_data = heart_data.tail()
shape_data = heart_data.shape
heart_data_isnull = heart_data.isnull().sum()






print(heart_data.isnull().sum())
print(heart_data.describe())






print(heart_data['target'].value_counts())

X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']
print(X)
print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)







model = LogisticRegression()
model.fit(X_train, Y_train)


X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)













app = Flask(__name__)

@app.route("/")
def home(): 
   return render_template("index.html",variable = result)

    
@app.route('/heart_logo.png')
def logo():
    return send_from_directory(app.static_folder, 'heart_logo.png')

@app.route("/buttonClicked", methods=["POST"])
def buttonClicked():
    global result
    input_data = request.get_json()
    age = input_data[0]
    gender = input_data[1]
    cp = input_data[2]
    trestbps = input_data[3]
    chol = input_data[4]
    fbs = input_data[5]
    restecg = input_data[6]
    thalach = input_data[7]
    exang = input_data[8]
    oldpeak = input_data[9]
    slope = input_data[10]
    ca = input_data[11]
    thal = input_data[12]
    input_data = (age,gender,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal)
    input_data_as_numpy_array= np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction = model.predict(input_data_reshaped)
    
    
    if (prediction[0]== 0):
        result =  " does not have a Heart Disease"
        print(result)
    else:
        result = " has Heart Disease"
        print(result)

    print("clicked") 
    return result
    

@app.route('/details', methods=['POST'])
def details():
    return render_template('details.html' ,head_data=head_data, tail_data=tail_data, 
    shape_data=shape_data,heart_data_isnull=heart_data_isnull,training_data_accuracy=training_data_accuracy,test_data_accuracy=test_data_accuracy)





if __name__ == "__main__":
    app.run(port=8000)


