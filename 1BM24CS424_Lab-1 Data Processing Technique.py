df = pd.read_csv("housing.csv")

print("Column Information:")
df.info()

print("\nStatistical Information (Numerical Columns):")
print(df.describe())

print("\nCount of Unique Labels in 'Ocean Proximity':")
print(df["Ocean Proximity"].value_counts())

print("\nColumns with Missing Values:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])




from google.colab import files
files.upload()


import os
os.listdir()



import pandas as pd

df = pd.read_csv("Dataset of Diabetes .csv")

print("----- Dataset Information -----")
df.info()

print("\n----- Statistical Information -----")
print(df.describe())

print("\n----- Columns with Missing Values -----")
missing = df.isnull().sum()
print(missing[missing > 0])


OUTPUT


----- Dataset Information -----
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1000 entries, 0 to 999
Data columns (total 14 columns):
 #   Column     Non-Null Count  Dtype  
---  ------     --------------  -----  
 0   ID         1000 non-null   int64  
 1   No_Pation  1000 non-null   int64  
 2   Gender     1000 non-null   object
 3   AGE        1000 non-null   int64  
 4   Urea       1000 non-null   float64
 5   Cr         1000 non-null   int64  
 6   HbA1c      1000 non-null   float64
 7   Chol       1000 non-null   float64
 8   TG         1000 non-null   float64
 9   HDL        1000 non-null   float64
 10  LDL        1000 non-null   float64
 11  VLDL       1000 non-null   float64
 12  BMI        1000 non-null   float64
 13  CLASS      1000 non-null   object
dtypes: float64(8), int64(4), object(2)
memory usage: 109.5+ KB

----- Statistical Information -----
                ID     No_Pation          AGE         Urea           Cr  \
count  1000.000000  1.000000e+03  1000.000000  1000.000000  1000.000000  
mean    340.500000  2.705514e+05    53.528000     5.124743    68.943000  
std     240.397673  3.380758e+06     8.799241     2.935165    59.984747  
min       1.000000  1.230000e+02    20.000000     0.500000     6.000000  
25%     125.750000  2.406375e+04    51.000000     3.700000    48.000000  
50%     300.500000  3.439550e+04    55.000000     4.600000    60.000000  
75%     550.250000  4.538425e+04    59.000000     5.700000    73.000000  
max     800.000000  7.543566e+07    79.000000    38.900000   800.000000  

             HbA1c         Chol           TG          HDL          LDL  \
count  1000.000000  1000.000000  1000.000000  1000.000000  1000.000000  
mean      8.281160     4.862820     2.349610     1.204750     2.609790  
std       2.534003     1.301738     1.401176     0.660414     1.115102  
min       0.900000     0.000000     0.300000     0.200000     0.300000  
25%       6.500000     4.000000     1.500000     0.900000     1.800000  
50%       8.000000     4.800000     2.000000     1.100000     2.500000  
75%      10.200000     5.600000     2.900000     1.300000     3.300000  
max      16.000000    10.300000    13.800000     9.900000     9.900000  

              VLDL          BMI  
count  1000.000000  1000.000000  
mean      1.854700    29.578020  
std       3.663599     4.962388  
min       0.100000    19.000000  
25%       0.700000    26.000000  
50%       0.900000    30.000000  
75%       1.500000    33.000000  
max      35.000000    47.750000  

----- Columns with Missing Values -----
Series([], dtype: int64)
ADULT




import pandas as pd

)
adult_df = pd.read_csv("Adult.csv")


print("Columns in Adult Income dataset:")
print(adult_df.columns)

print("\nMissing values in each column:")
print(adult_df.isnull().sum())

print("\nFirst 5 rows of Adult Income dataset:")
print(adult_df.head())


OUTPUT

Columns in Adult Income dataset:
Index(['age', 'workclass', 'fnlwgt', 'education', 'educational-num',
       'marital-status', 'occupation', 'relationship', 'race', 'gender',
       'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
       'income'],
      dtype='object')

Missing values in each column:
age                0
workclass          0
fnlwgt             0
education          0
educational-num    0
marital-status     0
occupation         0
relationship       0
race               0
gender             0
capital-gain       0
capital-loss       0
hours-per-week     0
native-country     0
income             0
dtype: int64

First 5 rows of Adult Income dataset:
   age  workclass  fnlwgt     education  educational-num      marital-status  \
0   25    Private  226802          11th                7       Never-married  
1   38    Private   89814       HS-grad                9  Married-civ-spouse  
2   28  Local-gov  336951    Assoc-acdm               12  Married-civ-spouse  
3   44    Private  160323  Some-college               10  Married-civ-spouse  
4   18          ?  103497  Some-college               10       Never-married  

          occupation relationship   race  gender  capital-gain  capital-loss  \
0  Machine-op-inspct    Own-child  Black    Male             0             0  
1    Farming-fishing      Husband  White    Male             0             0  
2    Protective-serv      Husband  White    Male             0             0  
3  Machine-op-inspct      Husband  Black    Male          7688             0  
4                  ?    Own-child  White  Female             0             0  

   hours-per-week native-country income  
0              40  United-States  <=50K  
1              50  United-States  <=50K  
2              40  United-States   >50K  
3              40  United-States   >50K  
4              30  United-States  <=50K  





import pandas as pd

df = pd.read_csv("Adult.csv", na_values=['?'])

print(df.isnull().sum())

OUTPUT

age                   0
workclass          2799
fnlwgt                0
education             0
educational-num       0
marital-status        0
occupation         2809
relationship          0
race                  0
gender                0
capital-gain          0
capital-loss          0
hours-per-week        0
native-country      857
income                0
dtype: int64
