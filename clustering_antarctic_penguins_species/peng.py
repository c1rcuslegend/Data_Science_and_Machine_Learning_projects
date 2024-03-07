import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Loading the dataset
penguins_df = pd.read_csv("data/penguins.csv")

# Inspecting the dataset for potential outliers 
# and na values and removing them
sns.boxplot(data=penguins_df)  
plt.show()
plt.clf()
penguins_clean = penguins_df.dropna()
for col in penguins_df.columns:
    if col=="sex":
        continue
    Q1, Q3 = np.quantile(penguins_clean[col], 0.25), np.quantile(penguins_clean[col], 0.75)
    IQR = Q3 - Q1
    minn, maxx = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    check = (penguins_clean[col]>=minn) & (penguins_clean[col]<=maxx)
    penguins_clean = penguins_clean[check] 

# Pre-processing data using standard scaling 
# and creating dummy variables
df = pd.get_dummies(penguins_clean).drop("sex_.", axis=1)
scaler = StandardScaler()
features = scaler.fit_transform(df)
penguins_preprocessed = pd.DataFrame(data=features, columns=df.columns)

# Perform PCA on scaled data to reduce demensionality
# First, finding out the minimum number of components 
# to describe the database
pca = PCA()
pca.fit(penguins_preprocessed)
# Visualising explained_variance_ratio
exp_variance = pca.explained_variance_ratio_
cum_exp_variance = np.cumsum(exp_variance)
sns.barplot(x=penguins_preprocessed.columns, 
            y=exp_variance).set_title('Principal Component')
plt.tick_params(labelrotation=90)
plt.show()
plt.clf()
cum_exp_variance = np.cumsum(exp_variance)
sns.lineplot(cum_exp_variance)
plt.axhline(y=0.9,linestyle='--') 
plt.show()
plt.clf()
# Setting the n_components element
n_components=sum(exp_variance>0.1)
# Applying them in the final PCA and getting the new features
pca = PCA(n_components=n_components)
features_PCA = pca.fit_transform(penguins_preprocessed)

# Determining the optimal number of clusters
# by using elbow analysis
data = []
for k in range(1,11):
    kmeans=KMeans(n_clusters=k, random_state=42)
    kmeans.fit(features_PCA)
    data.append(kmeans.inertia_)
sns.lineplot(x=np.arange(1,11),y=data)
plt.show()
plt.clf()
n_clusters = 4

# Creating the final KMeans model and visualising the result
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(features_PCA)
X = features_PCA[: , 0]
Y = features_PCA[: , 1]
sns.scatterplot(x=X, y=Y, hue = kmeans.labels_)
centroids = kmeans.cluster_centers_
X_centr = centroids[: , 0]
Y_centr = centroids[: , 1]
sns.scatterplot(x=X_centr, y=Y_centr, marker="d", color="blue", s=70)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title(f'K-means Clustering (K={n_clusters})')
plt.legend()
plt.show()

# Creating a final statistical dataframe for each cluster
penguins_clean["label"] = kmeans.labels_
num_columns = [col for col in penguins_clean.columns.values if col!="sex"]
stat_penguins = penguins_clean[num_columns].groupby("label").mean()
print(stat_penguins)
    