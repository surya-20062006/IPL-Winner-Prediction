import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

final_df = pd.read_csv("data/final_df.csv")

X = final_df.drop('result', axis=1)
y = final_df['result']

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
                handle_unknown='ignore'
            ),
            ['batting_team', 'bowling_team']
        )
    ],
    remainder='passthrough'
)

pipe = Pipeline([
    ('step1', trf),
    ('step2', LogisticRegression())
])

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

print("Accuracy:")
print(accuracy_score(y_test, y_pred))