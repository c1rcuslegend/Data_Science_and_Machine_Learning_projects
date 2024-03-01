import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline

# Read track metadata with genre labels 
# and track metrics with features
tracks = pd.read_csv("datasets/fma-rock-vs-hiphop.csv")
echonest_metrics = pd.read_json("datasets/echonest-metrics.json",precise_float=True)

# Merge two dataframes
echo_tracks = echonest_metrics.merge(tracks[["track_id","genre_top"]], 
                                     how="inner", on="track_id")

# Inspect the resultant dataframe
# echo_tracks.info()

# Create a correlation matrix
corr_metrics = echo_tracks.drop("genre_top",axis=1).corr()
corr_metrics.style.background_gradient()
sns.heatmap(corr_metrics, annot = True)
plt.show()
plt.clf()

# Create features and labels
features = echo_tracks.drop(["genre_top","track_id"],axis=1).values
labels = echo_tracks["genre_top"].values

# Split our data into training and testing
train_features, test_features, \
    train_labels, test_labels = train_test_split(features, labels, test_size=0.25, random_state=10)

# Normalize the future data
scaler = StandardScaler()
scaled_train_features = scaler.fit_transform(train_features)
scaled_test_features = scaler.transform(test_features)

# Use principal component analysis to reduce dimensionality of data
pca = PCA()
pca.fit(scaled_train_features)
exp_variance = pca.explained_variance_ratio_
sns.barplot(x=np.arange(0,8), y=exp_variance).set_title('Principal Component #')
plt.show()
plt.clf()

# Use cumulative explained variance plot 
# to determine how many features are required 
# to explain, say, about 85% of the variance 
cum_exp_variance = np.cumsum(exp_variance)
sns.lineplot(cum_exp_variance)
plt.axhline(y=0.85,linestyle='--') 

# Use that features that can explain 85% 
# of the varience to reduce demensionality
pca = PCA(n_components=6, random_state=10)
train_pca = pca.fit_transform(scaled_train_features)
test_pca = pca.transform(scaled_test_features)

# Train a decision tree to classify genre
tree = DecisionTreeClassifier(random_state=10)
tree.fit(train_pca, train_labels)
pred_labels_tree = tree.predict(test_pca)

# Compare decision tree to a logistic regression
logreg = LogisticRegression(random_state=10)
logreg.fit(train_features, train_labels)
pred_labels_logit = logreg.predict(test_features)

# Create the classification report for both models
class_rep_tree = classification_report(test_labels, pred_labels_tree)
class_rep_log = classification_report(test_labels, pred_labels_logit)
print(f"Decision Tree: \n {class_rep_tree}")
print(f"Logistic Regression: \n {class_rep_log}")
print("________________________________________\n")

# Balance our data for greater performance, because
# we want to have as much hip-hop songs as the rock ones
# After that redefine the train and test set
hop_only = echo_tracks[echo_tracks["genre_top"]=="Hip-Hop"]
rock_only = echo_tracks[echo_tracks["genre_top"]=="Rock"]
rock_only = rock_only.sample(n=len(hop_only),random_state=10)
rock_hop_bal = pd.concat([rock_only, hop_only], axis=0)
features = rock_hop_bal.drop(["genre_top","track_id"],axis=1).values
labels = rock_hop_bal['genre_top']
train_features, test_features, \
  train_labels, test_labels = train_test_split(
      features, labels, test_size=0.25, random_state=10 )
train_pca = pca.fit_transform(scaler.fit_transform(train_features))
test_pca = pca.transform(scaler.transform(test_features))

# Change if balancing has improved model bias

# Train decision tree on the balanced data
tree = DecisionTreeClassifier(random_state=10)
tree.fit(train_pca, train_labels)
pred_labels_tree = tree.predict(test_pca)

# Train logistic regression on the balanced data
logreg = LogisticRegression()
logreg.fit(train_pca, train_labels)
pred_labels_logit = logreg.predict(test_pca)

class_rep_tree = classification_report(test_labels, pred_labels_tree)
class_rep_log = classification_report(test_labels, pred_labels_logit)
print(f"Decision Tree: \n {class_rep_tree}")
print(f"Logistic Regression: \n {class_rep_log}")
print("________________________________________\n")

# Using cross-validation to evaluate models
# Beforehand create the pipelines 
tree_pipe = Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=6)), 
                      ("tree", DecisionTreeClassifier(random_state=10))])
logreg_pipe = Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=6)), 
                        ("logreg", LogisticRegression(random_state=10))])

kf = KFold(n_splits=10)
tree_score = cross_val_score(tree_pipe, features, labels, cv=kf)
logit_score = cross_val_score(logreg_pipe, features, labels, cv=kf)

print("Decision Tree:", np.mean(tree_score), "Logistic Regression:", np.mean(logit_score))
print("________________________________________\n")
