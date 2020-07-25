# %% read data
import pandas as pd

train = pd.read_csv(
    "house-prices-advanced-regression-techniques/train.csv"
)
test = pd.read_csv(
    "house-prices-advanced-regression-techniques/test.csv"
)


# %% checkout out first few rows
train.head()


# %% checkout out dataframe info
train.info()


# %% describe the dataframe
train.describe(include="all")


# %% SalePrice distribution
import seaborn as sns

sns.distplot(train["SalePrice"])


# %% SalePrice distribution w.r.t CentralAir / OverallQual / BldgType / etc


# %% SalePrice distribution w.r.t YearBuilt / Neighborhood
import matplotlib.pyplot as plt
plt.figure(figsize=(15, 8))
sns.boxplot(data=train, x="YearBuilt", y="SalePrice")
ax.set_xticklabels(
    ax.get_xticklabels(), rotation=45, fontsize=5
)

# %%
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_log_error
import numpy as np


def evaluate(reg, x, y):
    pred = reg.predict(x)
    result = np.sqrt(mean_squared_log_error(y, pred))
    return f"RMSLE score: {result:.3f}"


dummy_reg = DummyRegressor()

dummy_selected_columns = ["MSSubClass"]
dummy_train_x = train[dummy_selected_columns]
dummy_train_y = train["SalePrice"]

dummy_reg.fit(dummy_train_x, dummy_train_y)
print("Training Set Performance")
print(evaluate(dummy_reg, dummy_train_x, dummy_train_y))

truth = pd.read_csv("truth_house_prices.csv")
dummy_test_x = test[dummy_selected_columns]
dummy_test_y = truth["SalePrice"]

print("Test Set Performance")
print(evaluate(dummy_reg, dummy_test_x, dummy_test_y))

print("Can you do better than a dummy regressor?")


# %% your solution to the regression problem
selected_columns = [
    "LotShape",
    "BldgType",
    "LotArea",
    "Neighborhood",
    "1stFlrSF",
    "2ndFlrSF",
]

train_x = train[selected_columns]

#%%
# massage the training data set


train_y = train["SalePrice"]


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

ct = ColumnTransformer(
    [
        (
             "onehot_encoding",
             OneHotEncoder(handle_unknown="ignore"),
             [
                 # string columns
                 "LotShape",
                 "BldgType",
                 "Neighborhood",
             ]
        )
    ],
    remainder="passthrough",
)

train_x = ct.fit_transform(train_x)
clf = LinearRegression() # regressor of your choice

clf.fit(train_x, train_y)


# %%
evaluate(clf, train_x, train_y)



# %%

# same massage for testing data set
test_x = test[selected_columns]
test_x = ct.transform(test_x)


# %%
test_y = truth["SalePrice"]
evaluate(clf, test_x, test_y)


# %%
