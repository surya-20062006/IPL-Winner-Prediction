import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

print("Current Directory:")
print(os.getcwd())

final_df = pd.read_csv("data/final_df.csv")

X = final_df.drop('result', axis=1)
y = final_df['result']

print("\nX Shape :", X.shape)
print("y Shape :", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

trf = ColumnTransformer(
    [
        (
            'trf',
            OneHotEncoder(
                sparse_output=False,
                handle_unknown='ignore'
            ),
            ['batting_team', 'bowling_team']
        )
    ],
    remainder='passthrough'
)

trf.fit(X_train)

encoded_data = trf.transform(X_train)

print(encoded_data.shape)

print("\nTraining Data")
print(X_train.shape)

print("\nTesting Data")
print(X_test.shape)