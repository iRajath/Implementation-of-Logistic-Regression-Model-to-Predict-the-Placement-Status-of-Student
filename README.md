# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.
5. Display the results. 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: S Rajath
RegisterNumber:  212224240127
*/
```

```py
import pandas as pd
df=pd.read_csv("placement_Data.csv")
df.head()

data1=df.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=45)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(solver="liblinear")
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report=classification_report(y_test,y_pred)
print(classification_report)

```

## Output:
### Placement Dataset:
![image](https://github.com/user-attachments/assets/1234ccf5-8739-46b0-9352-2daa3cb3ccef)

![image](https://github.com/user-attachments/assets/c1c40c9d-f3dc-4a59-8d83-98f439419996)

### Missing and Duplication of Data:

![image](https://github.com/user-attachments/assets/ee6e7fd9-a5b5-44ac-894c-2eafe15ebd61)

![image](https://github.com/user-attachments/assets/583370a6-0146-4377-9a2d-d47f21141dfc)

![image](https://github.com/user-attachments/assets/1d6d3491-9eb2-485d-86e1-798cd7346ca0)

![image](https://github.com/user-attachments/assets/0d1a995b-cb67-42e2-9282-a9229f32df69)

### Y-Prediction Value

![image](https://github.com/user-attachments/assets/5414c6bd-417c-4580-98ee-0a6188a25132)

### Accuracy

![image](https://github.com/user-attachments/assets/b7c44f4e-e6bd-4a37-a97c-689f4d640767)

### Confusion matrix
![image](https://github.com/user-attachments/assets/c5bbab00-adbe-4d2d-a78b-40a935b98d45)

### Classification report

![image](https://github.com/user-attachments/assets/8ec74e15-3a60-4d6e-ba41-a46ad77b2648)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
