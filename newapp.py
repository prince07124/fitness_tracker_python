import streamlit as st
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# --- Header ---
st.markdown("<p class='big-font'>Personal Fitness Tracker</p>", unsafe_allow_html=True)
st.write("Enter your details below to predict the **calories burned** during exercise!")
st.write("""Track Your Estimated Calorie Burn!""")

st.write("""Enter your details like `Age`, `Gender`, `BMI` , `Activity Level`, `Step Count`, and more to get insights into your predicted calories burned. Stay informed and optimize your fitness journey!
into this WebApp and then you will see the predicted value of kilocalories burned.""" )

# --- Styling ---
st.markdown("""
    <style>
        .big-font { font-size:30px !important; font-weight: bold; }
        .highlight { background-color: #3176D0; padding: 2px 8px; border-radius: 5px; }
        .stApp { background-color: #3176D0; }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
        /* Animated Gradient Background */
        @keyframes gradientBG {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        .stApp {
            background: linear-gradient(-45deg, #1a1a2e, #16213e, #0f3460, #533483);
            background-size: 400% 400%;
            animation: gradientBG 10s ease infinite;
            color: white;
        }

        .big-font { font-size:30px !important; font-weight: bold; }
        .highlight { background-color:#3176D0; padding: 2px 8px; border-radius: 5px; }
    </style>
""", unsafe_allow_html=True)




# --- Sidebar Inputs ---
st.sidebar.header("User Input Parameters")
def user_input_features():
    age = st.sidebar.slider("Age:", 10, 100, 30)
    height = st.sidebar.slider("Height (cm):", 100, 220, 170)
    weight = st.sidebar.slider("Weight (kg):", 30, 150, 70)
    steps = st.sidebar.slider("Step Count:", 0, 20000, 5000)
    
    # Auto Calculate BMI
    bmi = round(weight / ((height / 100) ** 2), 2)
    st.sidebar.markdown(f"**BMI:** <span style='background-color: #FFDD57; padding: 2px 8px; border-radius: 5px;'>{bmi}</span>", unsafe_allow_html=True)
    
    duration = st.sidebar.slider("Duration (min):", 0, 70, 30)
    heart_rate = st.sidebar.slider("Heart Rate:", 60, 130, 80)
    body_temp = st.sidebar.slider("Body Temperature (°C):", 36, 42, 38)
    gender = st.sidebar.radio("Gender:", ("Male", "Female"))
    
    # Ensure gender encoding matches dataset
    gender_male = 1 if gender == "Male" else 0
    gender_female = 1 if gender == "Female" else 0

    return pd.DataFrame({
        "Age": [age], "Height": [height], "Weight": [weight], "BMI": [bmi], "Step_Count": [steps],
        "Duration": [duration], "Heart_Rate": [heart_rate], "Body_Temp": [body_temp],
        "Gender_Male": [gender_male], "Gender_Female": [gender_female]
    })

df = user_input_features()

# --- Show User Inputs ---
st.write("---")
st.subheader("Your Parameters:")
st.write(df)

# --- Load and Process Data ---
calories = pd.read_csv("calories.csv")
exercise = pd.read_csv("exercise.csv")
exercise_df = exercise.merge(calories, on="User_ID").drop(columns="User_ID")

# Add BMI Column
exercise_df["BMI"] = round(exercise_df["Weight"] / ((exercise_df["Height"] / 100) ** 2), 2)

# One-hot encode gender
exercise_df = pd.get_dummies(exercise_df, columns=["Gender"], drop_first=False)

# Add Step Count Column (if not present in dataset, initialize randomly)
if "Step_Count" not in exercise_df.columns:
    np.random.seed(1)
    exercise_df["Step_Count"] = np.random.randint(1000, 20000, size=len(exercise_df))

# Prepare Training Data
train_data, test_data = train_test_split(exercise_df, test_size=0.2, random_state=1)
X_train = train_data.drop("Calories", axis=1)
y_train = train_data["Calories"]
X_test = test_data.drop("Calories", axis=1)
y_test = test_data["Calories"]

# --- Train Model ---
model = RandomForestRegressor(n_estimators=1000, max_depth=6, max_features=3)
model.fit(X_train, y_train)

df = df.reindex(columns=X_train.columns, fill_value=0)
prediction = model.predict(df)[0]

# --- Display Prediction ---
st.write("---")
st.subheader("🔥 Predicted Calories Burned:")
st.markdown(f"<p style='background-color: #FFDD57; padding: 2px 8px; border-radius: 5px; font-size: 20px; font-weight: bold;'>{round(prediction, 2)} kilocalories</p>", unsafe_allow_html=True)

# --- Find Similar Results ---
similar_data = exercise_df[(exercise_df["Calories"] >= prediction - 10) & (exercise_df["Calories"] <= prediction + 10)]
st.write("---")
st.subheader("📊 Similar Results:")
st.write(similar_data.sample(5))

# --- Additional Stats ---
st.write("---")
st.subheader("📌 Insights Compared to Others:")
st.write(f"You are older than **`{round((exercise_df['Age'] < df['Age'][0]).mean() * 100, 2)}%`** of people.")
st.write(f"Your exercise duration is longer than **`{round((exercise_df['Duration'] < df['Duration'][0]).mean() * 100, 2)}%`** of people.")
st.write(f"Your heart rate is higher than **`{round((exercise_df['Heart_Rate'] < df['Heart_Rate'][0]).mean() * 100, 2)}%`** of people.")
st.write(f"Your body temperature is higher than **`{round((exercise_df['Body_Temp'] < df['Body_Temp'][0]).mean() * 100, 2)}%`** of people.")
st.write(f"Your step count is higher than **`{round((exercise_df['Step_Count'] < df['Step_Count'][0]).mean() * 100, 2)}%`** of people.")
