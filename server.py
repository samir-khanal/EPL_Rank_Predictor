from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib


app = Flask(__name__)

# Load trained models and scalers
regressor = joblib.load('random_forest_regressor.pkl')
regressor_pts_only = joblib.load('random_forest_regressor_pts_only.pkl') # Trained with Points only
classifier = joblib.load('svm_classifier.pkl') # Load the SVM model
scaler = joblib.load('scaler.pkl')
scaler_pts_only = joblib.load('scaler_pts_only.pkl')  # Scaler for Points only

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        method = data.get('method')

        # Checking if required keys exists or not
        if not 'method':
            return jsonify({"error": "Missing 'method' parameter"}), 400
        
        # Expecting a list of dictionaries for the team stats
        prediction_method = data.get('data')

        if not prediction_method or not isinstance(prediction_method, list):
            return jsonify({"error": "Missing or invalid 'data' parameter; it must be a list of dictionaries."}), 400

        # Work directly with the dictionary (first record)
        stats = prediction_method[0]


        if method == "All Stats":
            required_keys = ['W', 'D', 'L', 'GF', 'GA', 'Pts', 'Sh']
            for key in required_keys:
                if key not in stats:
                    return jsonify({"error": f"Missing column: {key}"}), 400
            
            # Calculate Goal Difference (GD) and total matches
            GD = stats['GF'] - stats['GA']
            total_matches = stats['W'] + stats['D'] + stats['L']
            if total_matches != 38:
                return jsonify({"error": "Total matches (W + D + L) must be 38"}), 400
            #total_matches = input_data['W'][0] + input_data['D'][0] + input_data['L'][0]
            
            # Building the input as a list (order must be matching training order)
            input_data = [[stats['W'], stats['D'], stats['L'], stats['GF'], stats['GA'], stats['Pts'], stats['Sh'], GD]]
            input_data_scaled = scaler.transform(input_data)

            # Predict rank (Regression)
            predicted_rank_reg = round(regressor.predict(input_data_scaled)[0])

            # Predict rank category (Classification)
            predicted_rank_clf = classifier.predict(input_data_scaled)[0]

        elif method == "Points Only":
            if 'Pts' not in stats:
                return jsonify({"error": "Missing column: Pts"}), 400
          
            input_data = [[stats['Pts']]]
            # Scale using the points-only scaler
            input_data_scaled = scaler_pts_only.transform(input_data)
            # Predict rank (Regression) using Points only
            predicted_rank_reg = round(regressor_pts_only.predict(input_data_scaled)[0])
            predicted_rank_clf = "N/A (Classification requires all stats)"  # Classification not used in this case

        else:
            return jsonify({'error': 'Invalid prediction method'}), 400

        response = {
            "predicted_rank": predicted_rank_reg,
            "predicted_rank_category": predicted_rank_clf
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

#for postman
# {
#   "method": "Points Only",
#   "data": [
#     {
#       "Pts": 70
#     }
#   ]
# }
# and 
# {
#   "method": "All Stats",
#   "data": [
#     {
#       "W": 20,
#       "D": 10,
#       "L": 8,
#       "GF": 70,
#       "GA": 40,
#       "Pts": 70,
#       "Sh": 500
#     }
#   ]
# }
