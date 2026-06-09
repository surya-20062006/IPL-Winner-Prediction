import pickle
import pandas as pd

# Load trained model
pipe = pickle.load(open('models/model.pkl', 'rb'))

# Sample Input
data = pd.DataFrame({
    'batting_team': ['Chennai Super Kings'],
    'bowling_team': ['Mumbai Indians'],
    'runs_left': [50],
    'balls_left': [30],
    'wickets_left': [7],
    'target': [180],
    'current_run_rate': [8.6],
    'required_run_rate': [10.0]
})

# Prediction Probability
result = pipe.predict_proba(data)

print("Batting Team Win Probability :",
      round(result[0][1] * 100, 2), "%")

print("Bowling Team Win Probability :",
      round(result[0][0] * 100, 2), "%")