# %% read data
import pandas as pd

train = pd.read_csv("titanic/train.csv")
test = pd.read_csv("titanic/test.csv")


# %% checkout out first few rows
train.head()


# %% checkout out dataframe info
train.info()


# %% describe the dataframe
train.describe(include="all")


# %% visualize the dataset, starting with the Survied distribution
import seaborn as sns

sns.countplot(x="Survived", data=train)


# %% Survived w.r.t Pclass / Sex / Embarked ?
sns.countplot(x="Survived", hue="Pclass", data=train)


# %% Age distribution ?
sns.countplot(x="Survived", hue="Sex", data=train)


# %% Survived w.r.t Age distribution ?
sns.countplot(x="Survived", hue="Embarked", data=train)


# %% Age distribution ?
sns.distplot(train["Age"])


# %% Survived w.r.t SibSp / Parch  ?
sns.distplot(train["Age"],bins=7)


# %% Survived w.r.t Age distribution ?
import matplotlib.pyplot as plt

sns.distplot(train[train["Survived"]==1]["Age"], label="Survived")
sns.distplot(train[train["Survived"]==0]["Age"], label="Passed away")
plt.legend()


# %%
sns.countplot
#%%



# %% Dummy Classifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score


def evaluate(clf, x, y):
    pred = clf.predict(x)
    result = f1_score(y, pred)
    return f"F1 score: {result:.3f}"


dummy_clf = DummyClassifier(random_state=2020)

dummy_selected_columns = ["Pclass"]
dummy_train_x = train[dummy_selected_columns]
dummy_train_y = train["Survived"]

dummy_clf.fit(dummy_train_x, dummy_train_y)
print("Training Set Performance")
print(evaluate(dummy_clf, dummy_train_x, dummy_train_y))

truth = pd.read_csv("truth_titanic.csv")
dummy_test_x = test[dummy_selected_columns]
dummy_test_y = truth["Survived"]

print("Test Set Performance")
print(evaluate(dummy_clf, dummy_test_x, dummy_test_y))

print("Can you do better than a dummy classifier?")


# %% Your solution to this classification problem
from sklearn.tree import DecisionTreeClassifier


selected_columns = [
    "Age",
    "Fare",
    "Cabin",
    "Embarked",
]

train_x = train[selected_columns]
ave_age = train_x["Age"].mean()

train_x["Cabin"] = train_x["Cabin"].fillna("NA")
train_x["Embarked"] = train_x["Embarked"].fillna("NA")
train_x["Age"] = train_x["Age"].fillna(ave_age)
train_y = train["Survived"]


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer(
    [
        (
             "onehot_encoding",
             OneHotEncoder(handle_unknown="ignore"),
             ["Cabin", "Embarked"]
        )
    ],
    remainder="passthrough",
)
# %%
train_x = ct.fit_transform(train_x)
clf = DecisionTreeClassifier()

clf.fit(train_x, train_y)


# %%
evaluate(clf, train_x, train_y)

# %%
test_x = test[selected_columns]
test_x["Cabin"] = test_x["Cabin"].fillna("NA")
test_x["Embarked"] = test_x["Embarked"].fillna("NA")
test_x["Age"] = test_x["Age"].fillna(ave_age)
test_x["Fare"] = test_x["Fare"].fillna(train["Fare"].mean())

# %%
test_x = ct.transform(test_x)
pred = clf.predict(test_x)

# %%
test_y = truth["Survived"]
evaluate(clf, test_x, test_y)

# %%
