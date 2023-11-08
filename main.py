import pandas as pd
import sys
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, normalize
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor


## Predict APPROX_COST


# %%

def get_size(obj):
    return str(sys.getsizeof(obj) / 1000000) + "MB"


def transform_rate(val, delimiter: str = '/'):
    if type(val) != type(np.nan):
        _x, _y = list(map(np.float64, val.split(delimiter)))
        return _x / _y


# %%

data = pd.read_csv("D:/Datasets/Zomato/zomato.csv")
data.drop(data.columns[[0, 1, 2, 7, 14, 9, 10, 11, 13]], axis=1, inplace=True)
# %%
data['rate'] = data['rate'].replace('-', np.nan)
data['rate'] = data['rate'].replace('NEW', np.nan)
data['rate'] = data['rate'].apply(transform_rate)
d = {'Yes': 1, 'No': 0}
data['online_order'] = data['online_order'].map(d)
data['book_table'] = data['book_table'].map(d)
data['votes'] = MinMaxScaler().fit_transform(data['votes'].to_numpy().reshape((-1, 1)))
data['approx_cost(for two people)'] = data['approx_cost(for two people)'].str.replace(',', '')
# %%
data['rate'].fillna(0.0, inplace=True)
data['location'] = LabelEncoder().fit_transform(data['location'])
data['location'] = SimpleImputer(strategy='most_frequent').fit_transform(data['location'].to_numpy().reshape((-1, 1)))
data['approx_cost(for two people)'] = SimpleImputer(strategy='mean').fit_transform(
    data['approx_cost(for two people)'].to_numpy().reshape((-1, 1)))

data.drop_duplicates(inplace=True)
# %%
print(data.info())
print(data.isna().sum())
# %%

ct = ColumnTransformer(transformers=[
    ('oneHot_encoder', OneHotEncoder(sparse_output=False), ['location', 'listed_in(type)', 'listed_in(city)']),
], remainder='passthrough')

data = ct.fit_transform(data)

# %%

X, y = data[::, :-1], data[::, -1]
X = normalize(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True)

# %%

lr = LinearRegression(n_jobs=-1)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# %%

rf = RandomForestRegressor(n_jobs=20, n_estimators=200, verbose=2)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# %%

xgbr = XGBRegressor(verbosity=3, n_estimators=1000, device='cuda')
xgbr.fit(X_train, y_train)
y_pred_xgbr = xgbr.predict(X_test)

# %%

xtr = ExtraTreesRegressor(n_jobs=20, n_estimators=200, verbose=2)
xtr.fit(X_train, y_train)
y_pred_xtr = xtr.predict(X_test)

# %%
gbr = GradientBoostingRegressor(n_estimators=200, verbose=3, loss='huber')
gbr.fit(X_train, y_train)
y_pred_gbr = gbr.predict(X_test)

# %%

print("r2 - Linear Regression:", r2_score(y_test, y_pred_lr))
print("r2 - XGBoost Regressor:", r2_score(y_test, y_pred_xgbr))
print("r2 - Extra Trees Regressor:", r2_score(y_test, y_pred_xtr))
print("r2 - Random Forest Regressor:", r2_score(y_test, y_pred_rf))
print("r2 - Gradient Boosting Regressor:", r2_score(y_test, y_pred_gbr))
