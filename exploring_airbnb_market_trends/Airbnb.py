import numpy as np
import pandas as pd

df1 = pd.read_csv("./data/airbnb_price.csv", index_col="listing_id")
df2 = pd.read_table("./data/airbnb_last_review.tsv", index_col="listing_id")
df3 = pd.read_excel("./data/airbnb_room_type.xlsx", index_col="listing_id")

df_final = df1.join(df2.join(df3,on="listing_id"), on="listing_id")

df_final["last_review"] = pd.to_datetime(df_final["last_review"], format='%B %d %Y') 
first_review, last_review = df_final["last_review"].min(), df_final["last_review"].max()

df_final["room_type"] = df_final["room_type"].str.lower()
room_types_count = df_final[["room_type"]].value_counts()
private_rooms_count = room_types_count["private room"]

df_final["price"] = df_final["price"].str.strip("dollars")
df_final["price"] = df_final["price"].astype(int)
price_mean = round(df_final["price"].mean(),2)

review_dates = pd.DataFrame({"first_reviewed":[first_review],"last_reviewed":[last_review], "nb_private_rooms":[private_rooms_count], "avg_price":[price_mean]})
