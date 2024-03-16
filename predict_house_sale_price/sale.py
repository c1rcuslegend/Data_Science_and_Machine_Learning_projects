import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_squared_error as MSE

SEED = 42
rmse_dict = {}

# reading the database
df = pd.read_csv("house_sales.csv")

# cleaning the data and getting rid of na values
df["city"] = np.where(df["city"]=="--", "Unknown", df["city"])
temp_month = df["months_listed"].dropna()
repl_val = round(temp_month.mean(),1)
df["months_listed"] = df["months_listed"].fillna(repl_val)
df["house_type"] = np.where(df["house_type"]=="Det.", "Detached", df["house_type"])
df["house_type"] = np.where(df["house_type"]=="Semi", "Semi-detached", df["house_type"])
df["house_type"] = np.where(df["house_type"]=="Terr.", "Terraced", df["house_type"])
df["house_type"] = df["house_type"].astype("category")
df["city"] = df["city"].astype("category")
df["area"] = df["area"].str.replace(" sq.m.","").astype("float64")
df["sale_date"] = pd.to_datetime(df["sale_date"], format='%Y-%m-%d')

# splitting the data
X = df.drop(["sale_date","house_id"],axis=1)
y = df["sale_price"]
X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=0.2, random_state=SEED)
X_train.reset_index(inplace=True)
X_train.drop(["index"],axis=1)
X_test.reset_index(inplace=True)
X_test.drop(["index"],axis=1)

# encoding categorical features
# splitting the columns by type
categorical_cols = ["city","house_type"]
numerical_cols = ["months_listed", "bedrooms", "area"]
# initializing the One Hot Encoder
# encodeing the categorical features and concatenating with the numeric ones
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')  
X_train_encoded = pd.concat([X_train[numerical_cols], pd.DataFrame(encoder.fit_transform(X_train[categorical_cols]))], axis=1)
X_test_encoded = pd.concat([X_test[numerical_cols], pd.DataFrame(encoder.transform(X_test[categorical_cols]))], axis=1)
# making sure that columns have the same data type
X_train_encoded.columns = X_train_encoded.columns.astype(str)
X_test_encoded.columns = X_train_encoded.columns.astype(str)

# training Linear Regression model
linear_regressor = LinearRegression()
linear_regressor.fit(X_train_encoded, y_train)
y_pred_lr = linear_regressor.predict(X_test_encoded)
rmse_dict["lr"] = MSE(y_test, y_pred_lr)**(1/2)

# training Random Forest Regressor with GridSearchCV hyperparameter tuning
random_forest_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [4, 6, 8],
    'min_samples_split': [2, 5, 10]
}
random_forest_regressor = GridSearchCV(RandomForestRegressor(), 
        random_forest_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)  
random_forest_regressor.fit(X_train_encoded, y_train)
random_forest_best = random_forest_regressor.best_estimator_
y_pred_rf = random_forest_best.predict(X_test_encoded)
rmse_dict["rf"] = MSE(y_test, y_pred_rf)**(1/2)

# training Gradient Boosting Regressor with GridSearchCV hyperparameter tuning
gradient_boosting_grid = {
    'learning_rate': [0.1, 0.05, 0.01],
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 8]
}
gradient_boosting_regressor = GridSearchCV(GradientBoostingRegressor(), 
        gradient_boosting_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)  
gradient_boosting_regressor.fit(X_train_encoded, y_train)
gradient_boosting_best = gradient_boosting_regressor.best_estimator_
y_pred_gb = gradient_boosting_best.predict(X_test_encoded)
rmse_dict["gb"] = MSE(y_test, y_pred_gb)**(1/2)

# creating VotingRegressor ensemble using tuned models
regressor_ensemble = VotingRegressor(estimators=[
    ('linear_reg', linear_regressor),
    ('random_forest', random_forest_best),
    ('gradient_boosting', gradient_boosting_best)
])
regressor_ensemble.fit(X_train_encoded, y_train)
y_pred_vr = regressor_ensemble.predict(X_test_encoded)
rmse_dict["vr"] = MSE(y_test, y_pred_vr)**(1/2)

# evaluating the best model
for k,v in rmse_dict.items():
    print(f"rmse for {k} model is {v:.2f}")
best_model = regressor_ensemble












