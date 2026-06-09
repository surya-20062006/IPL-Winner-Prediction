import os
import pandas as pd

print("Current Directory:")
print(os.getcwd())

from sklearn.model_selection import train_test_split

# Load dataset
final_df = pd.read_csv("data/final_df.csv")

print(final_df.head())
print(final_df.shape)

# Features
X = final_df.drop('result', axis=1)

# Target
y = final_df['result']

print("\nX Shape :", X.shape)
print("y Shape :", y.shape)

from sklearn.model_selection import train_test_split

# Split Dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data")
print(X_train.shape)

print("\nTesting Data")
print(X_test.shape)