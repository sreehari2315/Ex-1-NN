
<H3>ENTER YOUR NAME : SREE HARI K </H3> 
<H3>ENTER YOUR REGISTER NO : 212223230212</H3>
<H3>EX. NO : 1</H3>
<H3>DATE : 10-03-2025</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
<h4>Import Libraries</h4>

```
from google.colab import files
import pandas as pd
import seaborn as sns
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy import stats
import numpy as np
```

<h4>Read the dataset</h4>

```
df=pd.read_csv("Churn_Modelling.csv")
df.head()
df.tail()
df.columns
```
<h4>Check the missing data</h4>

```
df.isnull().sum()
df.duplicated()
```

<h4>Assigning Y</h4>

```
y = df.iloc[:, -1].values
print(y)
```

<h4>Check for duplicates</h4>

```
df.duplicated()
```

<h4>Check for outliers</h4>

```
df.describe()
```

<h4>Dropping string values data from dataset</h4>

```
data = df.drop(['Surname', 'Geography','Gender'], axis=1)
```

<h4>Checking datasets after dropping string values data from dataset</h4>

```
data.head()
```

<h4>Normalize the dataset</h4>

```
scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(data))
print(df1)
```

<h4>Split the dataset</h4>

```
X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
print(X)
print(y)
```

<h4>Training and testing model</h4>

```
X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2)
print("X_train\n")
print(X_train)
print("\nLenght of X_train ",len(X_train))
print("\nX_test\n")
print(X_test)
print("\nLenght of X_test ",len(X_test))
```

## OUTPUT:

Data checking

![data checking](https://github.com/user-attachments/assets/46f4a96d-0e75-4216-b585-d05067879b8b)

Duplicates identification

![checking for Duplicate and NULL values](https://github.com/user-attachments/assets/6799da0c-544e-4c1d-96da-ec493190a185)

Values of 'Y'

![Values of Y](https://github.com/user-attachments/assets/71850056-8595-4a83-a24d-28281fe07a75)

Outliers

![outliers](https://github.com/user-attachments/assets/e40fc1a8-85ce-4fc8-ab03-3f94d03eddfa)

Checking datasets after dropping string values data from dataset

![Checking datasets after dropping](https://github.com/user-attachments/assets/64b4b466-adb1-46ab-9d53-a21908cce656)

Normalize the dataset

![Normalizatuon](https://github.com/user-attachments/assets/31114f34-2121-40f3-836d-cba0e4540b34)

Split the dataset

![Split the Data set](https://github.com/user-attachments/assets/cde62209-b651-4c12-95fc-a583d8a092b4)

Training the Model

![Training Model](https://github.com/user-attachments/assets/8135f383-2397-4b4a-8443-658a7b8a2b12)

Testing the Model

![Testing Model](https://github.com/user-attachments/assets/1a45a293-2469-43e4-85da-e0bb21880915)

## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.
