# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries.

2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.

3.Import LabelEncoder and encode the dataset.

4.Import LogisticRegression from sklearn and apply the model on the dataset.

5.Predict the values of array.

6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

7.Apply new unknown values
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Safeeq Fazil .A
RegisterNumber: 212222240086 
*/
```
```

import pandas as pd
data=pd.read_csv('/content/Placement_Data.csv')
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred


from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])


```

## Output:
### Placement Data:

![image](https://github.com/Safeeq-Fazil/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118680361/92aa65c4-b02d-4f7c-86a7-5baad30edb2a)

### Salary Data:

![image](https://github.com/Safeeq-Fazil/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118680361/8a824e97-0df6-48f9-b834-9faee4e4d1a9)

### Checking the null() function:

![image](https://github.com/Safeeq-Fazil/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118680361/2f4597be-4ee2-48a5-98d3-8c17538e02b3)

### Data Duplicate:

![image](https://github.com/Safeeq-Fazil/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118680361/32ff2364-2757-4a7b-952f-af6b64f5641c)

### Print Data:

![image](https://github.com/Safeeq-Fazil/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118680361/fe4dbf5e-fd89-4517-afcd-ef9f28543c70)

### Data-status:

![image](https://github.com/Safeeq-Fazil/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118680361/c4f04f92-e393-4f07-b6e8-e56850dd5ae7)


### y_prediction array:

![image](https://github.com/Safeeq-Fazil/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118680361/e180d184-16c2-4b55-9547-1d6b22a61a31)

### Accuracy value:

![image](https://github.com/Safeeq-Fazil/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118680361/39033397-abed-417d-956f-e4a4b7cb49f8)

### Confusion array:

![image](https://github.com/Safeeq-Fazil/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118680361/9ef122c1-2a38-47a1-a5ee-59396bda73ef)

### Classification report:

![image](https://github.com/Safeeq-Fazil/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118680361/15955798-b15d-46bc-88a8-fe043fa5866c)

### Prediction of LR:

![image](https://github.com/Safeeq-Fazil/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118680361/d956c618-eed4-40fb-95bf-a6397fb10041)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
