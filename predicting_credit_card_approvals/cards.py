import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import Lasso

# Load the dataset
cc_apps = pd.read_csv("cc_approvals.data", header=None) 

# Replace "?" with NaN and drop columns
cc_apps.columns = cc_apps.columns.astype(str)
cc_apps.replace("?", np.NaN, inplace=True)
cc_apps = cc_apps.drop(["11","13"], axis=1)
cc_apps["15"] = np.where(cc_apps["15"]=="+", 1, 0)

# Deal with NaN values
cc_apps["1"] = cc_apps["1"].astype(float)
cc_apps["1"].fillna(value=cc_apps["1"].mean(), inplace=True)
for col in cc_apps.columns:
    if cc_apps[col].isna().sum()==0:
        continue
    cc_apps[col].fillna(value=cc_apps[col].value_counts().idxmax(), inplace=True)

# preprocessing the data by encoding the categorical features
labels = cc_apps["15"].values
cc_apps_dummies = pd.get_dummies(cc_apps.drop(["15"], axis=1))
features = cc_apps_dummies.values
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.3, random_state=42)

# select best features by using Lasso Regression
# first get the best hyperparameter
params = {"alpha":np.arange(0.00001, 10, 500)}
kf=KFold(n_splits=5,shuffle=True, random_state=42)
lasso = Lasso(max_iter=10000)
lasso_cv=GridSearchCV(lasso, param_grid=params, cv=kf)
lasso_cv.fit(X_train, y_train)
best_lasso_param = lasso_cv.best_params_
#using Lasso Regressor to plot the best features
lasso1 = Lasso(alpha=best_lasso_param["alpha"])
lasso1.fit(X_train, y_train)
lasso1_coef = np.abs(lasso1.coef_)
names = cc_apps_dummies.columns
plt.figure(figsize=(15,6))
sns.barplot(x=names,y=lasso1_coef).set_title("Feature #")
plt.xticks(rotation=45)
plt.show()
plt.clf()
# segregate those features
feature_subset=np.array(names)[lasso1_coef>0.001]
new_cc_apps = cc_apps_dummies[cc_apps_dummies.columns.intersection(feature_subset)]
# again split the data
new_features = new_cc_apps.values
X_train, X_test, y_train, y_test = train_test_split(
    new_features, labels, test_size=0.33, random_state=42)

# rescale the training and testing features
scaler=MinMaxScaler(feature_range=(0, 1))
rescaledX_train = scaler.fit_transform(X_train)
rescaledX_test = scaler.transform(X_test)

# train a logistic regression classifier
logreg = LogisticRegression()
logreg.fit(rescaledX_train, y_train)
y_pred = logreg.predict(rescaledX_test)
print(confusion_matrix(y_test, y_pred))
print("__________________\n")

# perform hyperparameter tuning
params = {"tol":[0.01, 0.001, 0.0001], 
              "max_iter":[100, 150, 200]}
kf = KFold(n_splits=5, shuffle=True, random_state=42)
grid_model = GridSearchCV(estimator=logreg, param_grid=params, cv=kf) 
res = grid_model.fit(rescaledX_train, y_train)
print(f"Highest score {res.best_score_} using {res.best_params_}")
print("__________________\n")

# extract the best model with the best performance score
final_model = res.best_estimator_
print(f"Accuracy: {final_model.score(rescaledX_test, y_test)}")


