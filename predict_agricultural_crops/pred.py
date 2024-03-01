import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import f1_score

#import data
crops = pd.read_csv("soil_measures.csv")

#Checking data
print(crops.isna().sum().sort_values())
print(crops["crop"].nunique())
print(crops.dtypes)

#preparing data
crops["crop"] = crops["crop"].astype("category")
X = crops.drop("crop", axis=1)
y = crops["crop"]

#split into two datasets
X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)

#which feature has the highest score   
for col in crops.columns:
    if col=="crop":
        continue
    X_pred = X_test[[col]]
    lreg = LogisticRegression(max_iter=2000, multi_class="multinomial")
    lreg.fit(X_train[[col]],y_train)
    y_pred = lreg.predict(X_pred)
    score = f1_score(y_test, y_pred, average="weighted")
    print(f"Score for {col} is {score}")

#get rid of highly correlated features
corr = X.corr()
sns.heatmap(corr, annot=True)
plt.show()
X = X[["N","K","ph"]]

#again split into two datasets with new features
X_train, X_test, y_train, y_test = train_test_split(\
                X, y, test_size=0.2, random_state=42)

#Train the model and predict the results
log_reg = LogisticRegression(max_iter=2000, multi_class="multinomial")
log_reg.fit(X_train,y_train)
y_pred = log_reg.predict(X_test)
model_performance = f1_score(y_test, y_pred, average="weighted")
