import pandas as pd
import numpy as np
from statsmodels.formula.api import logit
from statsmodels.graphics.mosaicplot import mosaic

df = pd.read_csv("./car_insurance.csv")
df["outcome"] = df["outcome"].astype(int) 
table={}

for col in df.columns:
    if col in ["id", "outcome"]:
        continue
    #print(col)
    m = logit(f"outcome~{col}",data=df).fit()
    prediction = m.predict(df[col])
    outcomes = pd.DataFrame({"actual_response": df["outcome"],
                             "predicted_response": np.round(prediction)})
    conf_matrix = m.pred_table()
    mosaic(conf_matrix, title=col)
    TN, TP, FN, FP  = conf_matrix[0,0], conf_matrix[1,1], conf_matrix[1,0], conf_matrix[0,1]
    table[col] = (TN + TP) / (TN + TP + FN + FP)
    #print(outcomes.head())


maxx = max(table.values())
for k,v in table.items():
    if v==maxx:
        best_feature_df = pd.DataFrame({"best_feature":[k],"best_accuracy":[v]})
        break