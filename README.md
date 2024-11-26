# Data-mining-
1. Apply data cleaning techniques on any dataset (e.g. Chronic Kidney Disease dataset from 

UCI repository). Techniques may include handling missing values, outliers and inconsistent 

values. Also, a set of validation rules may be specified for the particular dataset and validation

checks performed.

2. Apply data pre-processing techniques such as standardization/normalization, 

transformation, aggregation, discretization/binarization, sampling etc. on any dataset

3. Apply simple K-means algorithm for clustering any dataset. Compare the performance 

of clusters by varying the algorithm parameters. For a given set of parameters, plot a 

line graph depicting MSE obtained after each iteration.

4. Perform partitioning, hierarchical, and density-based clustering algorithms on a 

downloaded dataset and evaluate the cluster quality by changing the algorithm's 

parameters. 

5. Use Naive bayes, K-nearest, and Decision tree classification algorithms to build 

classifiers on any two datasets. Pre-process the datasets using techniques specified in 

Q2. Compare the Accuracy, Precision, Recall and F1 measure reported for each dataset 

using the abovementioned classifiers under the following situations:

i. Using Holdout method (Random sampling):

a) Training set = 80% Test set = 20% 

b) Training set = 66.6% (2/3rd of total), Test set = 33.3%

ii. Using Cross-Validation:

a) 10-fold 

b) 5-fold

6. Use the Decision Tree classification algorithm to construct a classifier on two datasets. 

Evaluate the classifier's performance by performing ten-fold cross validation. Compare 

the performance with that of:

i. Bagging ensemble consisting of 3, 5, 7, 9 Decision tree classifiers

ii. Adaboost ensemble consisting of 3, 5, 7, 9 Decision tree classifiers


solutions 
Q1
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
df = sns.load_dataset("titanic")

# 1. Handling Missing Values
# Fill missing values in 'Age' with the median, as age can vary significantly
df['age'].fillna(df['age'].median(), inplace=True)

# Fill missing values in 'Embarked' with the most common value (mode)
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)

# Drop the 'Cabin' column as it has too many missing values
df.drop(columns=['deck'], inplace=True)

# 2. Handling Outliers
# Handle outliers in 'Fare' using the IQR method
Q1 = df['fare'].quantile(0.25)
Q3 = df['fare'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove outliers
df = df[(df['fare'] >= lower_bound) & (df['fare'] <= upper_bound)]

# 3. Dealing with Inconsistent Values
# Standardize categorical data: Ensure 'Sex' is lowercase
df['sex'] = df['sex'].str.lower()

# Check and correct any inconsistencies in 'Embarked'
df['embarked'] = df['embarked'].str.strip().str.upper()
# The valid values for 'embarked' are 'C', 'Q', and 'S'
df = df[df['embarked'].isin(['C', 'Q', 'S'])]

# 4. Applying Validation Rules
#  'Age' is within a reasonable range
invalid_age = df[(df['age'] < 0)]
df = df.drop(invalid_age.index)

# 'Fare' is non-negative
df = df[df['fare'] >= 0]

# Save the cleaned dataset
df.to_csv('titanic-cleaned.csv', index=False)

print("Data cleaning complete. Cleaned dataset saved as 'titanic-cleaned.csv'.")

Q2
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Binarizer
import seaborn as sns

df = sns.load_dataset("titanic")

# Dropping rows with missing valuesu
df.dropna(subset=['age', 'fare', 'embarked', 'sex'], inplace=True)

# 1. Standardization
scaler = StandardScaler()
df[['age', 'fare']] = scaler.fit_transform(df[['age', 'fare']])

# 2. Normalization
normalizer = MinMaxScaler()
df[['age', 'fare']] = normalizer.fit_transform(df[['age', 'fare']])

# 3. Transformation (Logarithmic transformation of 'Fare')
df['fare_log'] = np.log1p(df['fare'])  # Using log1p to handle zero values

# 4. Aggregation
agg_df = df.groupby(['pclass', 'sex'])[['age', 'fare']].mean().reset_index()

# 5. Discretization/Binarization
df['age_bin'] = pd.cut(df['age'], bins=4, labels=['Young', 'Middle-aged', 'Senior', 'Old'])

# Binarize the 'Fare' column: 0 if Fare is below median, else 1
binarizer = Binarizer(threshold=df['fare'].median())
df['fare_bin'] = binarizer.fit_transform(df[['fare']])

# 6. Sampling
# Random Sampling: Select 30% of the data randomly
random_sample = df.sample(frac=0.3, random_state=1)


# Display some of the processed data
print(df.head())
print("Aggregated Data:", agg_df.head())
print("Random Sample Data:", random_sample.head())
print("Stratified_Sample Data:", stratified_sample.head())

Q3
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Titanic dataset from seaborn
df = sns.load_dataset("titanic")

# Preprocessing
# Select relevant columns
columns_to_use = ['pclass', 'sex', 'age', 'fare', 'sibsp', 'parch']
data = df[columns_to_use].copy()

# Handle missing values
data['age'].fillna(data['age'].median(), inplace=True)
data['fare'].fillna(data['fare'].median(), inplace=True)

# Encode categorical variables
label_encoder = LabelEncoder()
data['sex'] = label_encoder.fit_transform(data['sex'])

# Normalize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# K-Means algorithm with MSE tracking
def kmeans_with_mse(data, n_clusters, max_iter=100):
    np.random.seed(42)
    n_samples, n_features = data.shape

    # Randomly initialize cluster centers
    centers = data[np.random.choice(n_samples, n_clusters, replace=False)]

    mse_per_iteration = []
    for _ in range(max_iter):
        # Assign each point to the nearest cluster
        distances = np.linalg.norm(data[:, np.newaxis] - centers, axis=2)
        labels = np.argmin(distances, axis=1)

        # Calculate Mean Squared Error (MSE)
        mse = np.mean(np.min(distances, axis=1) ** 2)
        mse_per_iteration.append(mse)

        # Update cluster centers
        new_centers = np.array([data[labels == i].mean(axis=0) for i in range(n_clusters)])

        # Check for convergence (if centers do not change)
        if np.allclose(centers, new_centers, atol=1e-6):
            break

        centers = new_centers

    return labels, centers, mse_per_iteration

# Apply K-means clustering
k = 3  # Number of clusters
labels, centers, mse_per_iteration = kmeans_with_mse(data_scaled, n_clusters=k)

# Plot MSE over iterations
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(mse_per_iteration) + 1), mse_per_iteration, marker='o')
plt.title(f'MSE Over Iterations for k={k}')
plt.xlabel('Iteration')
plt.ylabel('Mean Squared Error (MSE)')
plt.grid()
plt.show()

# Performance comparison with different k values
k_values = [2, 3, 4, 5]
mse_per_k = []

for k in k_values:
    _, _, mse_per_iteration = kmeans_with_mse(data_scaled, n_clusters=k)
    mse_per_k.append(mse_per_iteration[-1])

# Plot MSE vs. Number of Clusters
plt.figure(figsize=(8, 5))
plt.plot(k_values, mse_per_k, marker='o')
plt.title('MSE vs. Number of Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Mean Squared Error (MSE)')
plt.grid()
plt.show()

Q4
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Binarizer
import seaborn as sns

df = sns.load_dataset("titanic")

# Dropping rows with missing valuesu
df.dropna(subset=['age', 'fare', 'embarked', 'sex'], inplace=True)

# 1. Standardization
scaler = StandardScaler()
df[['age', 'fare']] = scaler.fit_transform(df[['age', 'fare']])

# 2. Normalization
normalizer = MinMaxScaler()
df[['age', 'fare']] = normalizer.fit_transform(df[['age', 'fare']])

# 3. Transformation (Logarithmic transformation of 'Fare')
df['fare_log'] = np.log1p(df['fare'])  # Using log1p to handle zero values

# 4. Aggregation
agg_df = df.groupby(['pclass', 'sex'])[['age', 'fare']].mean().reset_index()

# 5. Discretization/Binarization
df['age_bin'] = pd.cut(df['age'], bins=4, labels=['Young', 'Middle-aged', 'Senior', 'Old'])

# Binarize the 'Fare' column: 0 if Fare is below median, else 1
binarizer = Binarizer(threshold=df['fare'].median())
df['fare_bin'] = binarizer.fit_transform(df[['fare']])

# 6. Sampling
# Random Sampling: Select 30% of the data randomly
random_sample = df.sample(frac=0.3, random_state=1)


# Display some of the processed data
print(df.head())
print("Aggregated Data:", agg_df.head())
print("Random Sample Data:", random_sample.head())
print("Stratified_Sample Data:", stratified_sample.head())

Q5
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.datasets import load_iris
import seaborn as sns

# Function to evaluate performance
def evaluate_model(y_test, y_pred, model_name, dataset_name):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"\n{model_name} on {dataset_name}")
    print(f"Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1-Score: {f1:.3f}")
    return accuracy, precision, recall, f1

# Dataset 1: Titanic Dataset
# Load Titanic dataset
df_titanic = sns.load_dataset("titanic")

# Preprocessing Titanic dataset
columns_to_use = ['pclass', 'sex', 'age', 'fare', 'sibsp', 'parch']
df_titanic = df_titanic[columns_to_use + ['survived']].dropna()
df_titanic['sex'] = LabelEncoder().fit_transform(df_titanic['sex'])
df_titanic['age'].fillna(df_titanic['age'].median(), inplace=True)
df_titanic['fare'].fillna(df_titanic['fare'].median(), inplace=True)

X_titanic = df_titanic.drop('survived', axis=1)
y_titanic = df_titanic['survived']

# Dataset 2: Iris Dataset
iris = load_iris()
X_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
y_iris = iris.target

# Standardize both datasets
scaler = StandardScaler()
X_titanic = scaler.fit_transform(X_titanic)
X_iris = scaler.fit_transform(X_iris)

# Define models
models = {
    "Naive Bayes": GaussianNB(),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

# Holdout Method
holdout_splits = {"80/20": 0.8, "66.6/33.3": 0.666}
results = []

for split_name, split_ratio in holdout_splits.items():
    print(f"\n--- Holdout Method ({split_name}) ---")
    for dataset_name, (X, y) in {"Titanic": (X_titanic, y_titanic), "Iris": (X_iris, y_iris)}.items():
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-split_ratio, random_state=42)
        for model_name, model in models.items():
            # Train and predict
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            # Evaluate
            metrics = evaluate_model(y_test, y_pred, model_name, dataset_name)
            results.append((model_name, dataset_name, split_name, *metrics))

# Cross-Validation
cross_val_folds = {"10-fold": 10, "5-fold": 5}

print("\n--- Cross-Validation ---")
for fold_name, folds in cross_val_folds.items():
    for dataset_name, (X, y) in {"Titanic": (X_titanic, y_titanic), "Iris": (X_iris, y_iris)}.items():
        for model_name, model in models.items():
            scores = cross_validate(model, X, y, cv=folds, scoring=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'])
            print(f"\n{model_name} on {dataset_name} ({fold_name})")
            print(f"Accuracy: {np.mean(scores['test_accuracy']):.3f}, Precision: {np.mean(scores['test_precision_weighted']):.3f}, Recall: {np.mean(scores['test_recall_weighted']):.3f}, F1-Score: {np.mean(scores['test_f1_weighted']):.3f}")
Q6 

import numpy as np

import pandas as pd

from sklearn.datasets import load_iris, load_wine

from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier



# Load datasets

iris = load_iris()

wine = load_wine()



# Define datasets

datasets = {

    "Iris": (iris.data, iris.target),

    "Wine": (wine.data, wine.target)

}



# Results dictionary to store performance metrics

results = {}



# Iterate over datasets

for dataset_name, (X, y) in datasets.items():

    dataset_results = {}



    # 1. Decision Tree Classifier

    dt = DecisionTreeClassifier(random_state=42)

    dt_scores = cross_val_score(dt, X, y, cv=10, scoring='accuracy')

    dataset_results["Decision Tree"] = np.mean(dt_scores)

    

    # 2. Bagging Classifier with different numbers of base estimators

    bagging_results = {}

    for n_estimators in [3, 5, 7, 9]:

        bagging = BaggingClassifier(base_estimator=DecisionTreeClassifier(random_state=42), 

                                    n_estimators=n_estimators, random_state=42)

        bagging_scores = cross_val_score(bagging, X, y, cv=10, scoring='accuracy')

        bagging_results[f"{n_estimators} estimators"] = np.mean(bagging_scores)

    dataset_results["Bagging"] = bagging_results

    

    # 3. AdaBoost Classifier with different numbers of base estimators

    adaboost_results = {}

    for n_estimators in [3, 5, 7, 9]:

        adaboost = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(random_state=42), 

                                      n_estimators=n_estimators, random_state=42)

        adaboost_scores = cross_val_score(adaboost, X, y, cv=10, scoring='accuracy')

        adaboost_results
