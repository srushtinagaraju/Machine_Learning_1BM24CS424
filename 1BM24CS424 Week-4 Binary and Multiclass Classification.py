#########BINARY CLASSIFICATION############
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



df = pd.read_csv("HR_comma_sep.csv")

print("First 5 Rows:\n", df.head())
print("\nDataset Shape:", df.shape)
print("\nColumns:\n", df.columns)



# Retention Count
print("\nEmployee Retention Count:\n", df['left'].value_counts())

sns.countplot(x='left', data=df)
plt.title("Employee Retention (0 = Stayed, 1 = Left)")
plt.show()




salary_retention = pd.crosstab(df['salary'], df['left'])
salary_retention.plot(kind='bar')

plt.title("Salary vs Employee Retention")
plt.xlabel("Salary Level")
plt.ylabel("Number of Employees")
plt.xticks(rotation=0)
plt.show()


# Some datasets use 'sales' instead of 'Department'
if 'Department' in df.columns:
    dept_col = 'Department'
else:
    dept_col = 'sales'

dept_retention = pd.crosstab(df[dept_col], df['left'])
dept_retention.plot(kind='bar', figsize=(10,6))

plt.title("Department vs Employee Retention")
plt.xlabel("Department")
plt.ylabel("Number of Employees")
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# Convert categorical variables into dummy variables
df = pd.get_dummies(df, columns=['salary', dept_col], drop_first=True)

# Select important features (based on EDA)
X = df[['satisfaction_level',
        'average_montly_hours',
        'promotion_last_5years',
        'time_spend_company',
        'salary_low',
        'salary_medium']]

y = df['left']



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)



model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)



y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()



################MULTICLASS CLASSIFICATION #############################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from google.colab import files
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


print("Please upload:")
print("1. zoo-data.csv (main dataset)")
print("2. zoo-class-type.csv (class info)")

uploaded = files.upload()

all_files = os.listdir()

# Pick the latest file containing 'zoo-data'
zoo_file_candidates = [f for f in all_files if "zoo-data" in f]
if len(zoo_file_candidates) == 0:
    raise FileNotFoundError("Could not find any file containing 'zoo-data'")
zoo_file = sorted(zoo_file_candidates)[-1]  # pick last uploaded

# Pick the latest file containing 'zoo-class-type'
class_file_candidates = [f for f in all_files if "zoo-class-type" in f]
if len(class_file_candidates) == 0:
    raise FileNotFoundError("Could not find any file containing 'zoo-class-type'")
class_file = sorted(class_file_candidates)[-1]  # pick last uploaded

print(f"\nUsing Zoo Data file: {zoo_file}")
print(f"Using Class Type file: {class_file}\n")

zoo_df = pd.read_csv(zoo_file)
class_df = pd.read_csv(class_file)

print("Zoo Data Sample:\n", zoo_df.head())
print("\nClass Type Details:\n", class_df)

if 'animal_name' in zoo_df.columns:
    zoo_df = zoo_df.drop('animal_name', axis=1)

X = zoo_df.drop('class_type', axis=1)
y = zoo_df['class_type']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


model = LogisticRegression(max_iter=2000, multi_class='multinomial')
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", round(accuracy*100,2), "%")

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Zoo Classification")
plt.show()

