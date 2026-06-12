import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# --------------------------------
# LOAD DATA
# --------------------------------

final_df = pd.read_csv("data/final_df.csv")

# --------------------------------
# CLEAN DATA
# --------------------------------

final_df.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)

final_df.dropna(inplace=True)

# --------------------------------
# FEATURES & TARGET
# --------------------------------

X = final_df.drop("result", axis=1)
y = final_df["result"]

# --------------------------------
# TRAIN TEST SPLIT
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# --------------------------------
# ONE HOT ENCODING
# --------------------------------

trf = ColumnTransformer(
    transformers=[
        (
            "team_encoder",
            OneHotEncoder(handle_unknown="ignore"),
            ["batting_team", "bowling_team"]
        )
    ],
    remainder="passthrough"
)

# --------------------------------
# PIPELINE
# --------------------------------

pipe = Pipeline([
    ("encoder", trf),
    ("model", LogisticRegression(max_iter=1000))
])

# --------------------------------
# TRAIN MODEL
# --------------------------------

pipe.fit(X_train, y_train)

# --------------------------------
# PREDICTION
# --------------------------------

y_pred = pipe.predict(X_test)

# --------------------------------
# EVALUATION
# --------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("=" * 50)
print("MODEL ACCURACY :", round(accuracy * 100, 2), "%")
print("=" * 50)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# --------------------------------
# SAVE MODEL
# --------------------------------

pickle.dump(
    pipe,
    open("models/model.pkl", "wb")
)

print("\n✅ Model Saved Successfully!")