import streamlit as st
import pandas as pd
import pickle
import time
import base64

# ---------------------------------
# PAGE CONFIG
# ---------------------------------

st.set_page_config(
    page_title="IPL Winner Prediction",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------
# LOAD CSS
# ---------------------------------

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def load_css():
    try:
        img_base64 = get_base64_of_bin_file("assets/stadium.jpg")
    except Exception:
        img_base64 = ""
        
    with open("style.css") as f:
        css = f.read().replace('url("assets/stadium.jpg")', f'url("data:image/jpeg;base64,{img_base64}")')
        st.markdown(
            f"<style>{css}</style>",
            unsafe_allow_html=True
        )

load_css()

# ---------------------------------
# TEAMS & LOGOS
# ---------------------------------

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

logos = {
    "Chennai Super Kings": "assets/csk.png",
    "Mumbai Indians": "assets/mi.png",
    "Royal Challengers Bangalore": "assets/rcb.png",
    "Kolkata Knight Riders": "assets/kkr.png",
    "Delhi Capitals": "assets/dc.png",
    "Punjab Kings": "assets/pbks.png",
    "Rajasthan Royals": "assets/rr.png",
    "Sunrisers Hyderabad": "assets/srh.png"
}

# ---------------------------------
# LOAD MODEL
# ---------------------------------

pipe = pickle.load(
    open(
        "models/model.pkl",
        "rb"
    )
)






# ---------------------------------
# HERO SECTION
# ---------------------------------

st.markdown(
"""
<div class="hero">

<h1>
🏏 IPL Winner Prediction
</h1>

<h3>
AI Powered Live Match Predictor
</h3>

</div>
""",
unsafe_allow_html=True
)

st.markdown("---")

#---------------------------------
# TEAM SELECTION
# ---------------------------------

col1, col2 = st.columns(2)

with col1:

    batting_team = st.selectbox(
        "🏏 Batting Team",
        sorted(teams)
    )

with col2:

    bowling_team = st.selectbox(
        "🎯 Bowling Team",
        sorted(teams)
    )

# ---------------------------------
# TEAM LOGOS DISPLAY
# ---------------------------------

logo1, logo2 = st.columns(2)

with logo1:

    st.image(
        logos[batting_team],
        width=160
    )

with logo2:

    st.image(
        logos[bowling_team],
        width=160
    )

st.markdown("---")


# ---------------------------------
# MATCH INPUTS
# ---------------------------------

st.markdown(
"""
<div class="card">
<h2>📊 Match Details</h2>
</div>
""",
unsafe_allow_html=True
)

c1, c2 = st.columns(2)

with c1:

    target = st.number_input(
        "🎯 Target Score",
        min_value=1,
        step=1
    )

with c2:

    score = st.number_input(
        "🏏 Current Score",
        min_value=0,
        step=1
    )

c3, c4 = st.columns(2)

with c3:

    overs = st.number_input(
        "⏱ Overs Completed",
        min_value=0.0,
        max_value=20.0,
        step=0.1
    )

with c4:

    wickets = st.number_input(
        "❌ Wickets Out",
        min_value=0,
        max_value=10,
        step=1
    )

st.markdown("---")

# ---------------------------------
# PREDICT BUTTON
# ---------------------------------

predict = st.button(
    "🔮 Predict Winner"
)

st.markdown("---")

# ---------------------------------
# MODEL PREDICTION
# ---------------------------------

if predict:

    if batting_team == bowling_team:
        st.error(
            "Batting Team and Bowling Team cannot be the same."
        )
        st.stop()

    if score > target:
        st.error(
            "Current Score cannot be greater than Target."
        )
        st.stop()

    over = int(overs)
    ball = int((overs - over) * 10)

    if ball > 5:
        st.error(
            "Use overs like 15.3"
        )
        st.stop()

    balls_bowled = over * 6 + ball
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
        required_run_rate = (
            runs_left * 6
        ) / balls_left

    input_df = pd.DataFrame({

        "batting_team":[batting_team],
        "bowling_team":[bowling_team],
        "runs_left":[runs_left],
        "balls_left":[balls_left],
        "wickets_left":[wickets_left],
        "target":[target],
        "current_run_rate":[current_run_rate],
        "required_run_rate":[required_run_rate]

    })

    result = pipe.predict_proba(
        input_df
    )

    loss = result[0][0]
    win = result[0][1]

    # -----------------------------
    # MATCH DASHBOARD
    # -----------------------------


    st.markdown(
    """
    <div class='card'>
    <h2>📊 Match Situation</h2>
    </div>
    """,
    unsafe_allow_html=True
    )

    m1,m2,m3,m4,m5=st.columns(5)

    m1.metric(
        "🎯 Target",
        target
    )

    m2.metric(
        "🏏 Score",
        f"{score}/{wickets}"
    )

    m3.metric(
        "⏱ Overs",
        overs
    )

    m4.metric(
        "🔥 Runs Left",
        runs_left
    )

    m5.metric(
        "⚡ Balls Left",
        balls_left
    )

            

    st.markdown("---")

        # -----------------------------
        # WIN PROBABILITY
        # -----------------------------

    st.markdown(
        """
        <div class='card'>
        <h2>🏆 Winning Probability</h2>
        </div>
        """,
        unsafe_allow_html=True
        )

    p1,p2 = st.columns(2)

    with p1:

        st.image(
            logos[batting_team],
            width=100
        )

        st.markdown(
            f"## {batting_team}"
        )

        st.progress(
            float(win)
        )

        st.metric(
            "Winning Chance",
            f"{round(win*100,2)}%"
        )

    with p2:

        st.image(
            logos[bowling_team],
            width=100
        )

        st.markdown(
            f"## {bowling_team}"
        )

        st.progress(
            float(loss)
        )

        st.metric(
            "Winning Chance",
            f"{round(loss*100,2)}%"
        )

    st.markdown("---")

    st.markdown(
        """
    <h1 style='text-align:center'>
    🏆
    </h1>
    """,
        unsafe_allow_html=True
    )

    # -----------------------------
    # EXTRA INFO
    # -----------------------------

    st.info(
        f"""
    Current Run Rate : {round(current_run_rate,2)}

    Required Run Rate : {round(required_run_rate,2)}
    """
    )


st.markdown("---")

# ---------------------------------
# FOOTER
# ---------------------------------
st.markdown("---")

st.markdown(
"""
<div style="text-align:center;padding:20px">

<h2>🏏 IPL Winner Prediction</h2>

Machine Learning Based Live Match Predictor

<br>

⭐ Built with Python | Pandas | Scikit-Learn | Streamlit

<br>

👨‍💻 Developed by <b>Surya</b>

</div>
""",
unsafe_allow_html=True
)