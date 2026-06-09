
import streamlit as st
import pickle
import pandas as pd

# -----------------------------
# PAGE CONFIG (MUST BE FIRST)
# -----------------------------

st.set_page_config(
    page_title="IPL Winner Prediction",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------

st.markdown("""
<style>

#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

.main{
    background-color:#0E1117;
}

h1{
    text-align:center;
    color:white;
    font-size:50px;
}

h4{
    text-align:center;
    color:#B0B3B8;
}

div.stButton > button:first-child{
    background:linear-gradient(90deg,#FF512F,#DD2476);
    color:white;
    height:55px;
    width:100%;
    border-radius:12px;
    font-size:22px;
    font-weight:bold;
    border:none;
}

[data-testid="metric-container"]{
    background-color:#1A1D24;
    border-radius:12px;
    padding:15px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODEL
# -----------------------------

pipe = pickle.load(open("models/model.pkl","rb"))

# -----------------------------
# SIDEBAR
# -----------------------------

st.sidebar.title("🏏 IPL Winner Prediction")

st.sidebar.markdown("---")

st.sidebar.success("""
### Machine Learning Project

Algorithm
- Logistic Regression

Accuracy
- 80.27%

Tech Stack
- Python
- Pandas
- Scikit-Learn
- Streamlit
""")

st.sidebar.markdown("---")

st.sidebar.info("""
Developer

👨‍💻 Surya
""")

# -----------------------------
# HEADER
# -----------------------------

st.markdown("""
<h1>🏏 IPL Winner Prediction</h1>
<h4>AI Powered Live Match Predictor</h4>
""", unsafe_allow_html=True)

# -----------------------------
# TEAMS
# -----------------------------

teams = [
    "Chennai Super Kings",
    "Mumbai Indians",
    "Royal Challengers Bangalore",
    "Kolkata Knight Riders",
    "Delhi Capitals",
    "Punjab Kings",
    "Rajasthan Royals",
    "Sunrisers Hyderabad"
]

# -----------------------------
# INPUT
# -----------------------------

col1,col2 = st.columns(2)

with col1:
    batting_team = st.selectbox(
        "Batting Team",
        sorted(teams)
    )

with col2:
    bowling_team = st.selectbox(
        "Bowling Team",
        sorted(teams)
    )

col3,col4 = st.columns(2)

with col3:
    target = st.number_input(
        "Target Score",
        min_value=1
    )

with col4:
    score = st.number_input(
        "Current Score",
        min_value=0
    )

col5,col6 = st.columns(2)

with col5:
    overs = st.number_input(
        "Overs Completed",
        min_value=0.0,
        max_value=20.0,
        step=0.1
    )

with col6:
    wickets = st.number_input(
        "Wickets Out",
        min_value=0,
        max_value=10
    )

# -----------------------------
# PREDICTION
# -----------------------------

if st.button("🔮 Predict Probability"):

    if batting_team == bowling_team:
        st.error("Both teams cannot be the same.")
        st.stop()

    if score > target:
        st.error("Current score cannot exceed target.")
        st.stop()

    over = int(overs)
    ball = int((overs-over)*10)

    if ball > 5:
        st.error("Use overs like 15.3")
        st.stop()

    balls_bowled = over*6 + ball
    balls_left = 120 - balls_bowled

    runs_left = target - score
    wickets_left = 10 - wickets

    if overs == 0:
        current_run_rate = 0
    else:
        current_run_rate = score / overs

    if balls_left == 0:
        required_run_rate = 0
    else:
        required_run_rate = (runs_left*6)/balls_left

    input_df = pd.DataFrame({
        'batting_team':[batting_team],
        'bowling_team':[bowling_team],
        'runs_left':[runs_left],
        'balls_left':[balls_left],
        'wickets_left':[wickets_left],
        'target':[target],
        'current_run_rate':[current_run_rate],
        'required_run_rate':[required_run_rate]
    })

    result = pipe.predict_proba(input_df)

    loss = result[0][0]
    win = result[0][1]

    st.balloons()

    st.success("Prediction Completed!")

    st.divider()

    st.subheader("📊 Match Situation")

    a,b,c = st.columns(3)

    with a:
        st.metric("Target",target)

    with b:
        st.metric("Score",f"{score}/{wickets}")

    with c:
        st.metric("Overs",overs)

    d,e = st.columns(2)

    with d:
        st.metric("Runs Left",runs_left)

    with e:
        st.metric("Balls Left",balls_left)

    st.divider()

    st.subheader("🏆 Winning Probability")

    x,y = st.columns(2)

    with x:

        st.markdown(f"### 🔥 {batting_team}")

        st.progress(win)

        st.metric(
            "Winning Chance",
            f"{round(win*100,2)}%"
        )

    with y:

        st.markdown(f"### ⚔️ {bowling_team}")

        st.progress(loss)

        st.metric(
            "Winning Chance",
            f"{round(loss*100,2)}%"
        )

st.divider()

st.markdown(
"""
<div style='text-align:center;'>

⭐ Built with Python, Pandas, Scikit-Learn & Streamlit

👨‍💻 Developed by <b>Surya</b>

</div>
""",
unsafe_allow_html=True
)
