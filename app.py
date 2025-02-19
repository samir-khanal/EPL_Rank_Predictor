import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Loading the saved models and scaler
regressor = joblib.load('random_forest_regressor.pkl')
regressor_pts_only = joblib.load('random_forest_regressor_pts_only.pkl')  # Trained with Points only
classifier = joblib.load('random_forest_classifier.pkl')
scaler = joblib.load('scaler.pkl')
scaler_pts_only = joblib.load('scaler_pts_only.pkl')  # Scaler for Points only

# Sidebar rules and help section
st.sidebar.title("About the App")
st.sidebar.markdown("""
### How to Use
1. Input the **team stats** in the form.
2. Select the **prediction method**:
   - **All Stats**: Uses all the team performance stats to predict rank.
   - **Points Only**: Uses only the total points (Pts) for rank prediction.
3. The app will predict:
   - The **rank** of the team using regression.
   - The **rank category** (e.g., Top 4, Top 8) using classification.
   
### Help Section
- **W**: Number of wins.
- **D**: Number of draws.
- **L**: Number of losses.
- **GF**: Goals scored (Goals For).
- **GA**: Goals conceded (Goals Against).
- **Pts**: Total points earned.
- **Sh**: Total shots taken.
- **GD**: Goal difference (GF - GA).
- **Total Matches**: Sum of Wins (W), Draws (D), and Losses (L). Must equal 38.
- **Predictions**:
   - **Regression**: Predicts the exact rank based on team performance.
   - **Classification**: Groups the team into rank categories (e.g., Top 4, Mid-Table).
                    
### Features
- Automatically calculates **Goal Difference (GD)** and **Total Matches**.
- Ensures inputs are valid (e.g., total matches = 38).
- Provides interpretation of predictions.
- User-friendly interface.
""")

# App title
st.title("⚽Football Team Rank Prediction App")
st.write("This app predicts the **rank** and **rank category** of a epl football team based on its performance stats.")

# Input Form
st.header("Enter Team Stats📝")
with st.form("prediction_form"):
    W = st.number_input("Wins (W)", min_value=0, max_value=38, value=18, step=1)
    D = st.number_input("Draws (D)", min_value=0, max_value=38, value=5, step=1)
    L = st.number_input("Losses (L)", min_value=0, max_value=38, value=15, step=1)
    GF = st.number_input("Goals For (GF)", min_value=0, value=80, step=1)
    GA = st.number_input("Goals Against (GA)", min_value=0, value=40, step=1)
    Pts = st.number_input("Points (Pts)", min_value=0, max_value=120, value=80, step=1)
    Sh = st.number_input("Shots (Sh)", min_value=0, value=600, step=1)

    GD = GF - GA  # Automatically calculate GD
    total_matches = W + D + L  # Calculate total matches
    st.write(f"Goal Difference (GD) is automatically calculated as: **{GD}**")
    st.write(f"Total Matches (W + D + L): **{total_matches}**")

    # Option to select prediction method
    prediction_method = st.selectbox("Select Prediction Method", ["All Stats", "Points Only"])

    # Submit button
    submit_button = st.form_submit_button("Predict")
    
if submit_button:
     # Validation for total matches
    if total_matches != 38:
        st.error("🚫The sum of Wins (W), Draws (D), and Losses (L) must be 38. Please adjust the values.")
    elif GD > 0 and Pts == 0:
        st.error("🚫Points (Pts) cannot be 0 if Goal Difference (GD) is positive. Please adjust the values.")
    elif Sh < GF:
            st.error("🚫Shots (Sh) cannot be less than Goals For (GF). Please adjust the values.")
    else:
        if prediction_method == "All Stats":
            # Create input DataFrame
            input_data = pd.DataFrame({
                'W': [W],
                'D': [D],
                'L': [L],
                'GF': [GF],
                'GA': [GA],
                'Pts': [Pts],
                'Sh': [Sh],
                'GD': [GD]
            })

            # Scale input data
            input_data_scaled = scaler.transform(input_data)

            # Regression prediction
            predicted_rank_reg = regressor.predict(input_data_scaled)[0]
            predicted_rank_reg_rounded = round(predicted_rank_reg)
        else:
            if Sh < GF:
                st.error("🚫Shots (Sh) cannot be less than Goals For (GF). Please adjust the values.")
            else:
                # Create input DataFrame for Points only
                input_data = pd.DataFrame({
                    'Pts': [Pts],
                    # 'GF': [GF],
                    # 'GA': [GA],
                    # 'GD': [GD]
                })
                # Scale input data of points only
                input_data_scaled = scaler_pts_only.transform(input_data)

                # Regression prediction points only
                predicted_rank_reg = regressor_pts_only.predict(input_data_scaled)[0]
                predicted_rank_reg_rounded = round(predicted_rank_reg)
            
        # Classification prediction (always using all stats)
        input_data_scaled_all_stats = scaler.transform(pd.DataFrame({
            'W': [W],
            'D': [D],
            'L': [L],
            'GF': [GF],
            'GA': [GA],
            'Pts': [Pts],
            'Sh': [Sh],
            'GD': [GD]
        }))
        # Classification prediction
        predicted_rank_clf = classifier.predict(input_data_scaled_all_stats)[0]

        # Display predictions
        st.subheader("Predictions🔮")
        st.write(f"🏆**Predicted Rank:** {predicted_rank_reg_rounded}")
        st.write(f"📊**Predicted Team Standing Category:** {predicted_rank_clf}")

        # Provide a visual separation
        st.write("---")

        st.write("### Interpretation")
        st.markdown("""
        - **Predicted Rank (Regression)**: The exact ranking of the team in standings according to their stats.
        - **Predicted Rank Category (Classification)**: Shown in broader category (e.g Top 4, Mid-Table).
        """)

        # Visualization: Bar Chart of Input Stats
        st.write("### Input Stats Visualization📊")
        input_stats = pd.DataFrame({
        'Stats': ['Wins', 'Draws', 'Losses', 'Goals For', 'Goals Against', 'Points', 'Shots', 'Goal Difference'],
        'Values': [W, D, L, GF, GA, Pts, Sh, GD]
        })
        st.bar_chart(input_stats.set_index('Stats'))

# Footer
st.sidebar.write("---")
st.sidebar.write("Developed by Samir Khanal using Streamlit.")

