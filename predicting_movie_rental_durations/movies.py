import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso 
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.neighbors import KNeighborsRegressor

SEED = 9
mse_dict = {}

# Reading the database
df = pd.read_csv("rental_info.csv", parse_dates=["return_date","rental_date"])

# Preprocessing data before analysis 
# by adding dummy variables and additional columns
df["rental_length_days"] = (df["return_date"] - df["rental_date"]).dt.days
df["deleted_scenes"]=np.where(df["special_features"].str.contains("Deleted Scenes"),1,0)
df["behind_the_scenes"]=np.where(df["special_features"].str.contains("Behind the Scenes"),1,0)
df = df.drop(["special_features","rental_date","return_date"],axis=1)

# Building a heatmap to get rid of correlated variables
plt.figure(figsize=(15,6))
sns.heatmap(df.corr(),annot=True)
plt.show()
plt.clf()
df = df.drop(["amount_2","length_2","rental_rate_2"],axis=1)

# Splitting the data
features_names = df.drop("rental_length_days",axis=1)
features = features_names.values
labels = df["rental_length_days"]
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=SEED)

# Selecting best features by using Lasso Regression
# First getting the best hyperparameter
params = {"alpha":np.arange(0.00001, 10, 500)}
kf=KFold(n_splits=5, shuffle=True, random_state=SEED)
lasso = Lasso(max_iter=1000)
lasso_cv=GridSearchCV(lasso, param_grid=params, cv=kf)
lasso_cv.fit(X_train, y_train)
best_lasso_param = lasso_cv.best_params_
# Using Lasso Regressor to plot the best features
lasso1 = Lasso(alpha=best_lasso_param["alpha"])
lasso1.fit(X_train, y_train)
lasso1_coef = np.abs(lasso1.coef_)
names = features_names.columns
plt.figure(figsize=(15,6))
sns.barplot(x=names,y=lasso1_coef).set_title("Feature name")
plt.xticks(rotation=45)
plt.show()
plt.clf()
# Segregating those features
feature_subset=np.array(names)[lasso1_coef>0.1]
new_features_names = features_names[features_names.columns.intersection(feature_subset)]
# Again splitting the data
new_features = new_features_names.values
X_train, X_test, y_train, y_test = train_test_split(
    new_features, labels, test_size=0.2, random_state=SEED)

# Running Linear Regression model on lasso chosen features
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
mse_dict["mse_lr"] = MSE(y_test, y_pred_lr)

# Runing Random Forest model on lasso chosen features
# performing hyperparameter tuning
params = {'n_estimators': np.arange(1,301,1),
          'max_depth':np.arange(1,11,1)}
rf = RandomForestRegressor()
rand_search = RandomizedSearchCV(
    rf, params, cv=10, random_state=SEED, n_jobs=-1)
rand_search.fit(X_train, y_train)
best_params = rand_search.best_params_
# training the final model and computing mse
rf = RandomForestRegressor(n_estimators=best_params["n_estimators"], 
                           max_depth=best_params["max_depth"], 
                           random_state=SEED, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
mse_dict["mse_rf"] = MSE(y_test, y_pred_rf)

# Training the K Neighbours model
# Tuning the optimal number of neighbours 
# by using the elbow method
arr = []
for i in np.arange(1,22,1):
    knn = KNeighborsRegressor(n_neighbors=i)
    knn.fit(X_train, y_train)
    y_pred_temp_knn = knn.predict(X_test)
    arr.append(MSE(y_test,y_pred_temp_knn))
sns.lineplot(x=np.arange(1,22,1),y=arr,marker="o")
plt.xticks(np.arange(1,22,1))
plt.show()
plt.clf()
n_neighbors = 7
# Training the final model
knn = KNeighborsRegressor(n_neighbors=n_neighbors)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
mse_dict["mse_knn"] = MSE(y_test, y_pred_knn)
    
# Training the Voting Regressor for models 
clf = VotingRegressor(
    estimators=[('lr', lr), ('rf', rf), ('knn', knn)], n_jobs=-1)
clf.fit(X_train, y_train)
y_pred_clf = clf.predict(X_test)
mse_dict["mse_clf"] = MSE(y_test, y_pred_clf)

# Comparing MSE's of all models
for k,v in mse_dict.items():
    print(f"{k} is {v:.4f}")
best_model, best_mse = rf, mse_dict["mse_rf"]















 


