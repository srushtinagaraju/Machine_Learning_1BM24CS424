
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"
housing = pd.read_csv(url)

print("\nFIRST 10 ROWS:\n")
print(housing.head(10))

print("\nDATA TYPES:\n")
print(housing.dtypes)

print("\nBASIC STATISTICS:\n")
print(housing.describe())

plt.figure(figsize=(8,5))
plt.hist(housing["median_house_value"], bins=50)
plt.xlabel("Median House Value")
plt.ylabel("Frequency")
plt.title("Histogram of Median House Value")
plt.show()

plt.figure(figsize=(6,4))
plt.boxplot(housing["median_house_value"], vert=False)
plt.xlabel("Median House Value")
plt.title("Box Plot of Median House Value")
plt.show()

print("\nMISSING VALUES (%):\n")
print(housing.isnull().mean() * 100)

imputer = SimpleImputer(strategy="median")
housing["total_bedrooms"] = imputer.fit_transform(
    housing[["total_bedrooms"]]
)

housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
    labels=[1, 2, 3, 4, 5]
)

strat_train_set, strat_test_set = train_test_split(
    housing,
    test_size=0.2,
    stratify=housing["income_cat"],
    random_state=42
)

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

strat_train_set = pd.get_dummies(
    strat_train_set,
    columns=["ocean_proximity"],
    drop_first=True
)
strat_test_set = pd.get_dummies(
    strat_test_set,
    columns=["ocean_proximity"],
    drop_first=True
)
strat_train_set, strat_test_set = strat_train_set.align(
    strat_test_set,
    join="left",
    axis=1,
    fill_value=0
)


numeric_features = strat_train_set.select_dtypes(include=[np.number]).columns

minmax_scaler = MinMaxScaler(feature_range=(0, 1))

strat_train_set[numeric_features] = minmax_scaler.fit_transform(
    strat_train_set[numeric_features]
)

strat_test_set[numeric_features] = minmax_scaler.transform(
    strat_test_set[numeric_features]
)

corr_matrix = strat_train_set.corr()

plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix, cmap="coolwarm", center=0)
plt.title("Correlation Matrix Heatmap")
plt.show()

plt.figure(figsize=(7,5))
plt.scatter(
    strat_train_set["median_income"],
    strat_train_set["median_house_value"],
    alpha=0.3
)
plt.xlabel("Median Income")
plt.ylabel("Median House Value")
plt.title("Median Income vs Median House Value")
plt.show()

train_set, test_set = train_test_split(
    housing,
    test_size=0.2,
    random_state=42
)

train_set_numeric = train_set.select_dtypes(include=[np.number])

corr_target = (
    train_set_numeric
    .corr()["median_house_value"]
    .sort_values(ascending=False)
)

print("\nCORRELATION WITH MEDIAN HOUSE VALUE:\n")
print(corr_target)

print("\nTOP 3 POSITIVELY CORRELATED FEATURES:")
print(corr_target[1:4])

print("\nMOST NEGATIVELY CORRELATED FEATURE:")
print(corr_target.tail(1))

X = train_set.drop("median_house_value", axis=1)
y = train_set["median_house_value"]

X_test = test_set.drop("median_house_value", axis=1)
y_test = test_set["median_house_value"]

X = pd.get_dummies(X, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)
X, X_test = X.align(X_test, join="left", axis=1, fill_value=0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

lin_reg = LinearRegression()
lin_reg.fit(X_scaled, y)

y_pred = lin_reg.predict(X_test_scaled)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMODEL PERFORMANCE:")
print(f"RMSE: {rmse}")
print(f"MAE : {mae}")
print(f"R2  : {r2}")


OUTPUT
 
FIRST 10 ROWS:

   longitude  latitude  housing_median_age  total_rooms  total_bedrooms  \
0    -122.23     37.88                41.0        880.0           129.0  
1    -122.22     37.86                21.0       7099.0          1106.0  
2    -122.24     37.85                52.0       1467.0           190.0  
3    -122.25     37.85                52.0       1274.0           235.0  
4    -122.25     37.85                52.0       1627.0           280.0  
5    -122.25     37.85                52.0        919.0           213.0  
6    -122.25     37.84                52.0       2535.0           489.0  
7    -122.25     37.84                52.0       3104.0           687.0  
8    -122.26     37.84                42.0       2555.0           665.0  
9    -122.25     37.84                52.0       3549.0           707.0  

   population  households  median_income  median_house_value ocean_proximity  
0       322.0       126.0         8.3252            452600.0        NEAR BAY  
1      2401.0      1138.0         8.3014            358500.0        NEAR BAY  
2       496.0       177.0         7.2574            352100.0        NEAR BAY  
3       558.0       219.0         5.6431            341300.0        NEAR BAY  
4       565.0       259.0         3.8462            342200.0        NEAR BAY  
5       413.0       193.0         4.0368            269700.0        NEAR BAY  
6      1094.0       514.0         3.6591            299200.0        NEAR BAY  
7      1157.0       647.0         3.1200            241400.0        NEAR BAY  
8      1206.0       595.0         2.0804            226700.0        NEAR BAY  
9      1551.0       714.0         3.6912            261100.0        NEAR BAY  

DATA TYPES:

longitude             float64
latitude              float64
housing_median_age    float64
total_rooms           float64
total_bedrooms        float64
population            float64
households            float64
median_income         float64
median_house_value    float64
ocean_proximity        object
dtype: object

BASIC STATISTICS:

          longitude      latitude  housing_median_age   total_rooms  \
count  20640.000000  20640.000000        20640.000000  20640.000000  
mean    -119.569704     35.631861           28.639486   2635.763081  
std        2.003532      2.135952           12.585558   2181.615252  
min     -124.350000     32.540000            1.000000      2.000000  
25%     -121.800000     33.930000           18.000000   1447.750000  
50%     -118.490000     34.260000           29.000000   2127.000000  
75%     -118.010000     37.710000           37.000000   3148.000000  
max     -114.310000     41.950000           52.000000  39320.000000  

       total_bedrooms    population    households  median_income  \
count    20433.000000  20640.000000  20640.000000   20640.000000  
mean       537.870553   1425.476744    499.539680       3.870671  
std        421.385070   1132.462122    382.329753       1.899822  
min          1.000000      3.000000      1.000000       0.499900  
25%        296.000000    787.000000    280.000000       2.563400  
50%        435.000000   1166.000000    409.000000       3.534800  
75%        647.000000   1725.000000    605.000000       4.743250  
max       6445.000000  35682.000000   6082.000000      15.000100  

       median_house_value  
count        20640.000000  
mean        206855.816909  
std         115395.615874  
min          14999.000000  
25%         119600.000000  
50%         179700.000000  
75%         264725.000000  
max         500001.000000  

MISSING VALUES (%):

longitude             0.000000
latitude              0.000000
housing_median_age    0.000000
total_rooms           0.000000
total_bedrooms        1.002907
population            0.000000
households            0.000000
median_income         0.000000
median_house_value    0.000000
ocean_proximity       0.000000
dtype: float64

CORRELATION WITH MEDIAN HOUSE VALUE:

median_house_value    1.000000
median_income         0.690647
total_rooms           0.133989
housing_median_age    0.103706
households            0.063714
total_bedrooms        0.047980
population           -0.026032
longitude            -0.046349
latitude             -0.142983
Name: median_house_value, dtype: float64

TOP 3 POSITIVELY CORRELATED FEATURES:
median_income         0.690647
total_rooms           0.133989
housing_median_age    0.103706
Name: median_house_value, dtype: float64

MOST NEGATIVELY CORRELATED FEATURE:
latitude   -0.142983
Name: median_house_value, dtype: float64

MODEL PERFORMANCE:
RMSE: 69721.52303238414
MAE : 50403.619136339345
R2  : 0.6290401809913706
