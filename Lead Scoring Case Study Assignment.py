#!/usr/bin/env python
# coding: utf-8

# # 1. Data Understanding and Processing
# 

# In[1]:


### Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# ## 1. 1 Loading and exploring the dataset.
# 

# In[3]:


### Load the dataset
leads_data = pd.read_csv(r'C:\Users\GAGAN\Desktop\Leads.csv')

###  Display the first few rows of the dataset
print("First few rows of the dataset:")
print(leads_data.head())

### Display the structure and features of the dataset
print(leads_data.info())


# ## 1.2 Handling missing values and encoding categorical variables.

# In[10]:


# Handle missing values

# Replace 'Select' values in categorical variables with NaN
leads_data.replace('Select', pd.NA, inplace=True)

# For numerical features, we can use mean or median to impute missing values
leads_data.fillna(leads_data.median(numeric_only=True), inplace=True)

# For categorical features, we can impute missing values with the most frequent value
leads_data.fillna(leads_data.mode().iloc[0], inplace=True)

# Check if all missing values have been handled
print("\nMissing values after preprocessing:")
print(leads_data.isnull().sum())

# Encoding categorical variables using one-hot encoding
leads_data = pd.get_dummies(leads_data, drop_first=True)

# Display the updated dataset after encoding and feature engineering
print(leads_data.head())


# # 2. Exploratory Data Analysis (EDA):

# ## 2.1 Exploring the Distribution of Features:

# In[ ]:


### Summary statistics for numerical features
print(leads_data.describe())

### Distribution of categorical features
for column in leads_data.select_dtypes(include=['object']).columns:
    print(leads_data[column].value_counts())


####  2.1.1 Visualization of numerical features (histograms, box plots, etc.)
import matplotlib.pyplot as plt

#### Histograms of numerical features
leads_data.hist(figsize=(10, 10))
plt.show()

#### Box plots of numerical features
numerical_features = leads_data.select_dtypes(include=['float64', 'int64'])
numerical_features.boxplot(figsize=(10, 6))
plt.show()

####  2.1.2 Visualization of Categorical Features:
import seaborn as sns

#### Count plots of categorical features
categorical_features = leads_data.select_dtypes(include=['object'])
for column in categorical_features.columns:
    plt.figure(figsize=(8, 6))
    sns.countplot(x=column, data=leads_data)
    plt.xticks(rotation=45)
    plt.show()


# ##  2.2 Relationship Between Features:

# In[ ]:


#### Scatter plot of numerical features
sns.pairplot(numerical_features)
plt.show()

#### Correlation matrix heatmap
correlation_matrix = numerical_features.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.show()


# ## 2.3 Analysing the Distribution of the Target Variable 'Converted':

# In[ ]:


### Distribution of the target variable 'Converted'
print(leads_data['Converted'].value_counts())

### Visualization of the target variable distribution (bar plot)
plt.figure(figsize=(6, 4))
sns.countplot(x='Converted', data=leads_data)
plt.title('Distribution of Converted Leads')
plt.xlabel('Converted')
plt.ylabel('Count')
plt.show()

# Visualization of the target variable distribution (pie chart)
plt.figure(figsize=(6, 6))
leads_data['Converted'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['lightblue', 'lightgreen'])
plt.title('Distribution of Converted Leads')
plt.ylabel('')
plt.show()


# # 3. Model Building:

# ## 3.1 Splitting the Dataset:

# In[ ]:


from sklearn.model_selection import train_test_split

### Define features (X) and target variable (y)
X = leads_data.drop(columns=['Converted'])
y = leads_data['Converted']

### Split the dataset into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# ## 3.2 Building a Logistic Regression Model:

# In[ ]:


from sklearn.linear_model import LogisticRegression

### Initialize the logistic regression model
logistic_model = LogisticRegression()

### Fit the model on the training data
logistic_model.fit(X_train, y_train)


# # 4.Model Evaluation: 

# ## 4.1 Assessing Performance on Testing Data:

# In[ ]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

### Predicting on the testing data
y_pred = logistic_model.predict(X_test)

### Calculating evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

### Print evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


# ## 4.2 Analyzing the confusion matrix

# In[ ]:


from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


# ## 4.3 ROC Curve

# In[ ]:


# ROC Curve
y_pred_proba = logistic_model.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='orange', label='ROC Curve')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()

# AUC Score
auc_score = roc_auc_score(y_test, y_pred_proba)
print("AUC Score:", auc_score)


# # 5. Lead Scoring

# ## 5.1 Assigning Lead Scores:

# In[ ]:


# Predict probabilities of conversion for all leads
lead_scores = logistic_model.predict_proba(X)[:, 1]

# Scale the probabilities to a range of 0 to 100
lead_scores_scaled = lead_scores * 100

