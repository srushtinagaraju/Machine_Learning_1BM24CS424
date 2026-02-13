Linear Regression
#Q Predict canada's per capita income in year 2020. Use the data file canada_per_capita_income.csv file. If required, apply the necessary data processing steps. Using this build a regression model and predict the per capita income for canadian citizens in year 2020


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from google.colab import files


uploaded = files.upload()


df = pd.read_csv('canada_per_capita_income.csv')

print("Dataset Preview:")
print(df.head())

print("\nColumn Names:")
print(df.columns)
X = df[['year']]
y = df['per capita income (US$)']  

model = LinearRegression()
model.fit(X, y)

predicted_income = model.predict([[2020]])

print("\nPredicted Per Capita Income in 2020:", predicted_income[0])

plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.xlabel("Year")
plt.ylabel("Per Capita Income (US$)")
plt.title("Canada Income Prediction")
plt.show()


OUTPUT
canada_per_capita_income.csv
canada_per_capita_income.csv(text/csv) - 874 bytes, last modified: 2/13/2026 - 100% done
Saving canada_per_capita_income.csv to canada_per_capita_income (1).csv
Dataset Preview:
   year  per capita income (US$)
0  1970              3399.299037
1  1971              3768.297935
2  1972              4251.175484
3  1973              4804.463248
4  1974              5576.514583

Column Names:
Index(['year', 'per capita income (US$)'], dtype='object')




#Q  Predict Salary of the employee. Use the data file salary.csv file. If required, apply the necessary data processing steps. Using this build a regression model and predict the salary of the employee with 12 years of experience.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from google.colab import files

uploaded = files.upload()

df = pd.read_csv('salary.csv')

print("Dataset Preview:")
print(df.head())

print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

df = df.dropna()

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

X = df[['YearsExperience']]
y = df['Salary']

model = LinearRegression()
model.fit(X, y)

predicted_salary = model.predict([[12]])

print("\nPredicted Salary for 12 years experience:", predicted_salary[0])

plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Salary Prediction using Linear Regression")
plt.show()


OUTPUT
salary.csv
salary.csv(text/csv) - 346 bytes, last modified: 2/13/2026 - 100% done
Saving salary.csv to salary (1).csv
Dataset Preview:
   YearsExperience  Salary
0              1.1   39343
1              1.3   46205
2              1.5   37731
3              2.0   43525
4              2.2   39891

Missing Values Before Cleaning:
YearsExperience    2
Salary             0
dtype: int64

Missing Values After Cleaning:
YearsExperience    0
Salary             0
dtype: int64



Multiple Linear Regression
#Q Considering the data file hiring.csv. The file contains hiring statics for a firm such as experience of candidate, his written test score and personal interview score. Based on these 3 factors, HR will decide the salary. Given this data, you need to build a Multiple Linear Regression model for HR department that can help them decide salaries for future candidates. Using this predict salaries for following candidates,
#Q1 2 yr experience, 9 test score, 6 interview score
#Q2 12 yr experience, 10 test score, 10 interview score

import pandas as pd
from sklearn.linear_model import LinearRegression
from google.colab import files

uploaded = files.upload()

df = pd.read_csv('hiring (3).csv')  

word_to_num = {
    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
    'five': 5, 'six': 6, 'seven': 7, 'eight': 8,
    'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12
}

df['experience'] = df['experience'].replace(word_to_num)
df['experience'] = pd.to_numeric(df['experience'])

df['experience'] = df['experience'].fillna(0)
df['test_score(out of 10)'] = df['test_score(out of 10)'].fillna(
    df['test_score(out of 10)'].mean()
)

X = df[['experience', 'test_score(out of 10)', 'interview_score(out of 10)']]
y = df['salary($)']


model = LinearRegression()
model.fit(X, y)

new_candidates = pd.DataFrame({
    'experience': [2, 12],
    'test_score(out of 10)': [9, 10],
    'interview_score(out of 10)': [6, 10]
})

predictions = model.predict(new_candidates)

print("Salary for (2,9,6):", predictions[0])
print("Salary for (12,10,10):", predictions[1])



OUTPUT
Salary for (2,9,6): 53290.892559447646
Salary for (12,10,10): 92268.07227783566




#Q  Considering the data file 1000_companies.csv. The file contains profit statics for a firm such as R&D Spend, Administration, Marketing Spend and State. Based on these four factors build a Multiple Linear Regression model to predict the profit. Using this predict profit for following,
# Q1  91694.48  R&D Spend, 515841.3  Administration, 11931.24  Marketing Spend, Florida State

import pandas as pd
from sklearn.linear_model import LinearRegression
from google.colab import files

uploaded = files.upload()

df = pd.read_csv('1000_companies.csv')  

df = pd.get_dummies(df, columns=['State'], drop_first=True)

X = df.drop('Profit', axis=1)
y = df['Profit']

model.fit(X, y)


new_data = pd.DataFrame([{
    'R&D Spend': 91694.48,
    'Administration': 515841.3,
    'Marketing Spend': 11931.24,
    **{col: 0 for col in X.columns if 'State_' in col}
}])

predicted_profit = model.predict(new_data)

print("Predicted Profit for given data:", predicted_profit[0])


OUTPUT
Saving 1000_companies.csv to 1000_companies (1).csv
Predicted Profit for given data: 511017.34614637314
