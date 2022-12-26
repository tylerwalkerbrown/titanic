```python
import pandas as pd
import matplotlib as plt
import matplotlib.pyplot as plt

```

# Reading in Data


```python
df = pd.read_html('https://github.com/datasciencedojo/datasets/blob/master/titanic.csv')
```


```python
df = df[0]
```


```python
df = df.drop(['Unnamed: 0'], axis=1)
```

# Packages


```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

```

# Dropping Columns/ One-Hot Encoding


```python
#Selecting everyhting except target
x = df.loc[:, df.columns != 'Survived']
x = x.loc[:, x.columns != 'Name']
x = x.loc[:, x.columns != 'PassengerId']
x = x.loc[:, x.columns != 'Ticket']
x = x.loc[:, x.columns != 'Name']
x = x.loc[:, x.columns != 'Cabin']


#Grabbing the target variables
#y = variables.iloc[:, -1]
y = df.Survived
```


```python
x = pd.get_dummies(data = x, columns = ['Sex', 'Embarked'])
```


```python
x.isna().sum()
```




    Pclass        0
    Age           0
    SibSp         0
    Parch         0
    Fare          0
    Sex_female    0
    Sex_male      0
    Embarked_C    0
    Embarked_Q    0
    Embarked_S    0
    dtype: int64




```python
#Filling na values 
x['Embarked'] = x['Embarked'].fillna('S')
x['Age'] = x['Age'].fillna(x.Age.mean())
```

# Train Test Split 25% and Fitting using Standard Scalar


```python
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.25, random_state = 0)
```


```python
#Using standard scaler to improve the model performance
ss = StandardScaler()

X_sc_train = ss.fit_transform(x_train)
X_sc_test = ss.fit_transform(x_test)
```

# Cross Validating Every Model


```python
#LogisticRegression
lr = LogisticRegression()

pd.DataFrame(pd.DataFrame(cross_validate(lr, X_sc_train, y_train, return_train_score=True)).mean())
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>fit_time</th>
      <td>0.014262</td>
    </tr>
    <tr>
      <th>score_time</th>
      <td>0.000356</td>
    </tr>
    <tr>
      <th>test_score</th>
      <td>0.797958</td>
    </tr>
    <tr>
      <th>train_score</th>
      <td>0.802401</td>
    </tr>
  </tbody>
</table>
</div>




```python
#RandomForest
rfc = RandomForestClassifier(n_estimators=100)

pd.DataFrame(pd.DataFrame(cross_validate(rfc, X_sc_train, y_train, return_train_score=True)).mean())
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>fit_time</th>
      <td>0.086886</td>
    </tr>
    <tr>
      <th>score_time</th>
      <td>0.005589</td>
    </tr>
    <tr>
      <th>test_score</th>
      <td>0.806890</td>
    </tr>
    <tr>
      <th>train_score</th>
      <td>0.983531</td>
    </tr>
  </tbody>
</table>
</div>




```python
#GuassianNB
gnb = GaussianNB()

pd.DataFrame(pd.DataFrame(cross_validate(gnb, X_sc_train, y_train, return_train_score=True)).mean())
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>fit_time</th>
      <td>0.002143</td>
    </tr>
    <tr>
      <th>score_time</th>
      <td>0.000614</td>
    </tr>
    <tr>
      <th>test_score</th>
      <td>0.791976</td>
    </tr>
    <tr>
      <th>train_score</th>
      <td>0.793790</td>
    </tr>
  </tbody>
</table>
</div>




```python
#cross_validate
svc = SVC(kernel='linear')

pd.DataFrame(pd.DataFrame(cross_validate(svc, X_sc_train, y_train, return_train_score=True)).mean())
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>fit_time</th>
      <td>0.010167</td>
    </tr>
    <tr>
      <th>score_time</th>
      <td>0.001840</td>
    </tr>
    <tr>
      <th>test_score</th>
      <td>0.790439</td>
    </tr>
    <tr>
      <th>train_score</th>
      <td>0.790422</td>
    </tr>
  </tbody>
</table>
</div>




```python
#GradientBoostingClassifier
gbc = GradientBoostingClassifier()
pd.DataFrame(pd.DataFrame(cross_validate(gbc, X_sc_train, y_train, return_train_score=True)).mean())
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>fit_time</th>
      <td>0.053947</td>
    </tr>
    <tr>
      <th>score_time</th>
      <td>0.000486</td>
    </tr>
    <tr>
      <th>test_score</th>
      <td>0.815890</td>
    </tr>
    <tr>
      <th>train_score</th>
      <td>0.907934</td>
    </tr>
  </tbody>
</table>
</div>



# Fitting Model According to Crossvalidation Results


```python
#Assigning the classifier to gbc variable 
gbc.fit(x_train,y_train)
```




    GradientBoostingClassifier()




```python
#creating predictor set 
y_pred = gbc.predict(x_test)
```

# Assessing Results


```python
from sklearn import metrics
```


```python
print("Gradient Boost Classifier predicted", y_pred.sum(), "people surviving.")
print("The actual number of survivors was", y_test.sum(),".")
```

    Gradient Boost Classifier predicted 73 people surviving.
    The actual number of survivors was 84 .



```python
print(classification_report(y_test, y_pred))
```

                  precision    recall  f1-score   support
    
               0       0.85      0.91      0.88       139
               1       0.84      0.73      0.78        84
    
        accuracy                           0.84       223
       macro avg       0.84      0.82      0.83       223
    weighted avg       0.84      0.84      0.84       223
    


Assessing the accuracy of the model is when you take in the total number of correct predictions and divide by the count sum of observations.


```python
#Accuracy score
accuracy_score(y_test,y_pred)
```




    0.8430493273542601




```python
conf_matrix = metrics.confusion_matrix(y_test, y_pred)
conf_matrix
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = conf_matrix, display_labels = ['No WS', 'Won'])
cm_display.plot()
```




    <sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x7ff6754b31c0>




    
![png](output_30_1.png)
    

