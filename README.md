# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.model_selection import train_test_split
data = pd.read_csv(r"C:\Users\acer\Downloads\income(1) (1).csv")  
print(data.head())
print(data.info())
```

<img width="751" height="785" alt="Screenshot 2025-10-16 152419" src="https://github.com/user-attachments/assets/eb0252cd-0266-4221-8b62-e2a611187d8b" /> 

```
print(data.isnull().sum()) 
```

<img width="612" height="397" alt="Screenshot 2025-10-16 152602" src="https://github.com/user-attachments/assets/513face9-b49c-42a8-a35d-7b4b25d7d4bf" /> 

```
data = data.dropna()
categorical_cols = data.select_dtypes(include=['object']).columns
print( list(categorical_cols))
```

<img width="1412" height="49" alt="Screenshot 2025-10-16 152725" src="https://github.com/user-attachments/assets/15318404-ed30-4256-af64-0452e9cde54c" /> 

```
le = LabelEncoder()
for col in categorical_cols:
    data[col] = le.fit_transform(data[col])
print(data.head()) 
```

<img width="995" height="364" alt="Screenshot 2025-10-16 152838" src="https://github.com/user-attachments/assets/029a571d-7780-4d09-9dd1-7efd72f6aa17" /> 

```
target_col = data.columns[-1]
X = data.drop(columns=[target_col])
y = data[target_col]
print(X.head())
print(y.head()) 
```

<img width="1025" height="519" alt="Screenshot 2025-10-16 152949" src="https://github.com/user-attachments/assets/168cf2f2-ebb9-4db3-9338-be0044ea40e7" /> 

```
scaler_std = StandardScaler()
X_standardized = scaler_std.fit_transform(X)
X_standardized = pd.DataFrame(X_standardized, columns=X.columns)
print(X_standardized.head())
```

<img width="1085" height="361" alt="Screenshot 2025-10-16 153050" src="https://github.com/user-attachments/assets/37ba72e5-557c-4126-9d67-cf2660b8111c" /> 

```
scaler_mm = MinMaxScaler()
X_minmax = scaler_mm.fit_transform(X)
X_minmax = pd.DataFrame(X_minmax, columns=X.columns)
print(X_minmax.head()) 
```

<img width="1089" height="372" alt="Screenshot 2025-10-16 153220" src="https://github.com/user-attachments/assets/31f81764-2326-4df7-ad78-8803035e3316" />

```
scaler_robust = RobustScaler()
X_robust = scaler_robust.fit_transform(X)
X_robust = pd.DataFrame(X_robust, columns=X.columns)
print(X_robust.head())
```

<img width="1134" height="371" alt="Screenshot 2025-10-16 153330" src="https://github.com/user-attachments/assets/4391edf8-4d57-4e4b-b677-a167de694eec" /> 

```
X_train, X_test, y_train, y_test = train_test_split(X_standardized, y, test_size=0.3, random_state=42)
X_chi = X_minmax.copy()  # Chi2 requires non-negative values
chi2_selector = SelectKBest(score_func=chi2, k='all')
chi2_selector.fit(X_chi, y)
chi2_scores = pd.DataFrame({
    'Feature': X.columns,
    'Chi2 Score': chi2_selector.scores_
}).sort_values(by='Chi2 Score', ascending=False)

print(chi2_scores)
```

<img width="629" height="356" alt="Screenshot 2025-10-16 153512" src="https://github.com/user-attachments/assets/c41384a1-0438-4e97-a42d-f46a9e23af31" />

```
anova_selector = SelectKBest(score_func=f_classif, k='all')
anova_selector.fit(X_train, y_train)
anova_scores = pd.DataFrame({
    'Feature': X.columns,
    'F-Score': anova_selector.scores_
}).sort_values(by='F-Score', ascending=False)

print(anova_scores) 
```

<img width="902" height="358" alt="Screenshot 2025-10-16 153701" src="https://github.com/user-attachments/assets/bd8dc05e-5608-4b85-877f-a855fd616f3b" /> 

```
mi_selector = SelectKBest(score_func=mutual_info_classif, k='all')
mi_selector.fit(X_train, y_train)
mi_scores = pd.DataFrame({
    'Feature': X.columns,
    'Mutual Info Score': mi_selector.scores_
}).sort_values(by='Mutual Info Score', ascending=False)
print(mi_scores) 
```

<img width="870" height="379" alt="Screenshot 2025-10-16 153814" src="https://github.com/user-attachments/assets/fc0e159e-b996-45b2-8a60-e8c681a350ae" />





# RESULT:
       # INCLUDE YOUR RESULT HERE
