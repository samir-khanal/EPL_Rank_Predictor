import streamlit as st
import pandas as pd
import base64
import requests
import json

#(Ensure your Flask API is running at this address)
API_URL = "http://127.0.0.1:5000/predict"  # Change when deployed


# Sidebar rules and help section
st.sidebar.title("About the App")
st.sidebar.markdown("""
### How to Use
1. Select the **prediction method**:
   - **All Stats**: Uses all the team performance stats to predict rank.
   - **Points Only**: Uses only the total points (Pts) for rank prediction.
2. Input the **team stats** in the form.
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
""")
# Function to set background
def set_background(image_path):
    try:
        with open(image_path, "rb") as img_file:
            encoded_string = base64.b64encode(img_file.read()).decode()
        page_bg = f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        </style>
        """
        st.markdown(page_bg, unsafe_allow_html=True)
    except Exception as e:
        st.error("Error loading background image: " + str(e))

# Set EPL-themed background
set_background("epl.jpg")  # Ensure this image is in the same folder as your script
# App title
st.title("⚽Football Team Rank Predictor")
st.write("This app predicts the **rank** and **rank category** of a epl football team based on its performance stats.")

# Option to select prediction method
st.header("Select Prediction Method")
prediction_method = st.selectbox("Choose Prediction Method", ["All Stats", "Points Only"])

# Initialize variables
W, D, L, GF, GA, Pts, Sh, GD = 0, 0, 0, 0, 0, 0, 0, 0

# Input Form based on selected prediction method
st.header("Enter Team Stats📝")
with st.form("prediction_form"):
    if prediction_method == "All Stats":
        W = st.number_input("Wins (W)", min_value=0, max_value=38, value=10, step=1)
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
    else:
        Pts = st.number_input("Points (Pts)", min_value=0, max_value=114, value=80, step=1)
        st.write(f"Goal Difference (GD) is automatically calculated as: **{GD}**")

    # Submit button
    submit_button = st.form_submit_button("Predict")

# Initialize variables to avoid NameError
predicted_rank_reg_rounded = None
predicted_rank_clf = None

# Send request when button is clicked
if submit_button:
    if prediction_method == "All Stats":
        # Validation for total matches
        if total_matches != 38:
            st.error("🚫The sum of Wins (W), Draws (D), and Losses (L) must be 38.")
            st.stop()  # Stop further execution if validation fails
        elif GD > 0 and Pts == 0: 
            st.error("🚫Points (Pts) cannot be 0 if Goal Difference (GD) is positive. Please adjust the values.")
            st.stop() 
        elif Sh < GF: 
            st.error("🚫Shots (Sh) cannot be less than Goals For (GF). Please adjust the values.")
            st.stop()
        else:
            # Build the payload directly as a dictionary
            if prediction_method == "All Stats":
                payload = {
                    "method": prediction_method,
                    "data": [{
                        "W": W,
                        "D": D,
                        "L": L,
                        "GF": GF,
                        "GA": GA,
                        "Pts": Pts,
                        "Sh": Sh,
                        "GD": GD
                    }]
                }

            # Convert DataFrame to JSON
            #input_data_json = input_data.to_json(orient='records')
    # Build the payload for "Points Only"
    else:
        payload = {
            "method": prediction_method,
            "data": [{
                "Pts": Pts,
                "GD": GD
            }]
        }
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        result = response.json()    

        # Convert DataFrame to JSON
        #input_data_json = input_data.to_json(orient='records')
    
        predicted_rank_reg_rounded = result.get("predicted_rank")
        predicted_rank_clf = result.get("predicted_rank_category")

        # Display predictions
        st.subheader("Predictions🔮")
        st.write(f"🏆**Predicted Rank:** {predicted_rank_reg_rounded}")
        st.write(f"📊**Predicted Rank Category:** {predicted_rank_clf}")
    
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
    except Exception as e:
        st.error("API Request Failed: " + str(e))

# Reset button
if st.button("Reset Form"):
    st.rerun()

# Connect with me section with images
st.markdown(
    """
    ### Connect with me:
    [![GitHub](https://image-url/github.png)](https://github.com/samir-khanal?tab=repositories)

    [![LinkedIn](https://image-url/linkedin.png)](https://www.linkedin.com/in/samir-khanal-59569234a/)
    """,
    unsafe_allow_html=True,
)
# Footer
st.sidebar.write("---")
st.sidebar.write("Developed by Samir Khanal using Streamlit.")
