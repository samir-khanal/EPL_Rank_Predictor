import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, classification_report, accuracy_score
import numpy as np

df = pd.read_csv('seasonstats.csv')
# Add Goal Difference (GD) column
df['GD'] = df['GF'] - df['GA']
df['Rank'] = df.groupby('Season')['Pts'].rank(ascending=False, method='first')
# Create Classification Target
def classify_rank(rank):
    if rank <= 4:
        return "TOP 4"
    elif rank <= 8:
        return "TOP 8"
    elif rank <= 12:
        return "MID-TABLE"
    elif rank <= 17:
        return "LOWER MID-TABLE"
    else:
        return "Bottom 3(RELEGATION)"

df['Rank_Category'] = df['Rank'].apply(classify_rank)

df.head(20)
#df.drop(columns=["PK","SOT"], inplace=True)
df.info()
df.isnull().sum()
#print(f'The shape of the dataset is {df.shape}')

#df["Sh"] = df["Sh"].fillna(df["Sh"].mean()).round(2)
mean_value = df["Sh"].mean()

#function to fill NaN based on the season
def fill_sh(row):
    if row["Season"] < "1960/1961":  # Before 1960/1961
        return mean_value / 2 if pd.isna(row["Sh"]) else row["Sh"]
    else:  # 1960/1961 and later
        return mean_value if pd.isna(row["Sh"]) else row["Sh"]

# Apply the function to the "Sh" column
df["Sh"] = df.apply(fill_sh, axis=1)
df["Sh"] = df["Sh"].round(2)
#df.set_index("Season", inplace=True)
print(df)

#checking if the above code was executed correctly or not
print("Rows before 1960/1961:")
print(df[df["Season"] < "1960/1961"])

print("\nRows for 1960/1961 and later:")
print(df[df["Season"] >= "1960/1961"])

# Visualizations

# Visualize features vs rank
plt.figure(figsize=(14, 7))
sns.scatterplot(x='Pts', y='Rank', data=df, hue='Season')
plt.title('Points vs Rank')
plt.show()

plt.figure(figsize=(14, 7))
# Wins vs Rank
plt.subplot(2, 3, 1)
sns.scatterplot(x='W', y='Rank', data=df)
plt.title('Wins vs Rank')
plt.tight_layout()
plt.show()

# Compute and visualize the correlation heatmap
# Select only the numeric columns
numeric_cols = df.select_dtypes(include=[np.number])
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Features and target 
x = df[['W', 'D', 'L', 'GF', 'GA','Pts', 'Sh', 'GD']] #features
x_pts_only = df[['Pts']]  # Features for Points as main factor
y_reg = df['Rank']  # Target for regression
y_clf = df['Rank_Category']  # Target for classification


# Standardize features for all stats except points
scaler = StandardScaler() 
x_scaled = scaler.fit_transform(x) 

# Standardize features for Points Only
scaler_pts_only = StandardScaler()
x_pts_only_scaled = scaler_pts_only.fit_transform(x_pts_only)

# Add polynomial features to allow the model to capture non-linear relationships
# poly = PolynomialFeatures(degree=2)
# x_poly = poly.fit_transform(x_scaled)

#for regression of all stats
x_train_reg, x_test_reg, y_train_reg, y_test_reg = train_test_split(x_scaled, y_reg, test_size=0.20, random_state=42)

# Train-Test Split for Points Only
x_train_pts_only, x_test_pts_only, y_train_reg_pts_only, y_test_reg_pts_only = train_test_split(x_pts_only_scaled, y_reg, test_size=0.20, random_state=42)

#for classification
x_train_clf, x_test_clf, y_train_clf, y_test_clf = train_test_split(x_scaled, y_clf, test_size=0.20, random_state=42)

# training the regression model(all stats)
model_reg = RandomForestRegressor(random_state=42)
model_reg.fit(x_train_reg, y_train_reg)

# training the regression model (points only)
model_reg_pts_only = RandomForestRegressor(random_state=42)
model_reg_pts_only.fit(x_train_pts_only, y_train_reg_pts_only)

# prediction for regression (all stats)
y_pred_reg = model_reg.predict(x_test_reg)

# Evaluate regression model(all stats)
mse = mean_squared_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)
print(f"Regression Results:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Evaluate Regression Model (Points Only)
y_pred_reg_pts_only = model_reg_pts_only.predict(x_test_pts_only)
mse_pts_only = mean_squared_error(y_test_reg_pts_only, y_pred_reg_pts_only)
r2_pts_only = r2_score(y_test_reg_pts_only, y_pred_reg_pts_only)

print(f"\nRegression Results (Points Only):")
print(f"Mean Squared Error: {mse_pts_only:.2f}")
print(f"R² Score: {r2_pts_only:.2f}")

# Accuracy on training data(all stats)
train_accuracy = model_reg.score(x_train_reg, y_train_reg) * 100
print(f"Training Accuracy: {train_accuracy:.2f}%")

# Accuracy on training data(pts only)
train_accuracy = model_reg_pts_only.score(x_train_pts_only, y_train_reg_pts_only) * 100
print(f"Training Accuracy(Points only): {train_accuracy:.2f}%")

# Accuracy on test data(all stats)
test_accuracy = model_reg.score(x_test_reg, y_test_reg) * 100
print(f"Test Accuracy: {test_accuracy:.2f}%")

# Accuracy on test data(pts only)
test_accuracy = model_reg_pts_only.score(x_test_pts_only, y_test_reg_pts_only) * 100
print(f"Test Accuracy(Points only): {test_accuracy:.2f}%")

from sklearn.svm import SVC

# training classification model with SVM
model_clf = SVC(kernel='linear',random_state=42, class_weight='balanced')  # Default kernel is 'rbf'
model_clf.fit(x_train_clf, y_train_clf)

#predictions for classification
y_pred_clf = model_clf.predict(x_test_clf)

# Evaluate classification model
print(f"\n**Classification Results**")
print(f"Classification Report:\n{classification_report(y_test_clf, y_pred_clf)}")
classification_accuracy = accuracy_score(y_test_clf, y_pred_clf) * 100
print(f"Classification Accuracy: {classification_accuracy:.2f}%")

# Residual Plot
residuals = y_test_reg - y_pred_reg
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_pred_reg, y=residuals)
plt.axhline(0, color="red", linestyle="--")
plt.title("Residual Plot")
plt.xlabel("Predicted Rank")
plt.ylabel("Residuals")
plt.show()

# Predicted vs Actual Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test_reg, y=y_pred_reg, color="blue", label="Predictions")
plt.plot([min(y_test_reg), max(y_test_reg)], [min(y_test_reg), max(y_test_reg)], color="red", linestyle="--", label="Ideal Fit")
plt.title("Predicted vs Actual")
plt.xlabel("Actual Rank")
plt.ylabel("Predicted Rank")
plt.legend()
plt.show()

# Predict future league rankings for all stats
future_data = pd.DataFrame({
    'W': [10],
    'D': [5],
    'L': [15],
    'GF': [80],
    'GA': [40],
    'Pts': [80],
    'Sh': [600],
    'GD': [46]
})
future_data_scaled = scaler.transform(future_data) 
# Regression prediction
future_rank_reg = model_reg.predict(future_data_scaled)[0]
future_rank_reg_rounded = round(future_rank_reg)
print(f"\nPredicted Rank for Future Season (Regression): {future_rank_reg_rounded:.2f}")

# Classification prediction
future_rank_clf = model_clf.predict(future_data_scaled)[0]
print(f"Predicted Rank Category for Future Season (Classification): {future_rank_clf}")

# For Pts only
future_data_pts_only = pd.DataFrame({
    'Pts': [80],
    # 'GF': [80],
    # 'GA':[20],
    # 'GD':[60]
})
# Scale future data using the same scaler
future_data_pts_only_scaled = scaler_pts_only.transform(future_data_pts_only)

future_rank_pts_only = model_reg_pts_only.predict(future_data_pts_only_scaled)[0]
future_rank_pts_only_rounded = round(future_rank_pts_only)
print(f"Predicted Rank (Pts Only): {future_rank_pts_only_rounded:.2f}")

import joblib

# Save the regression model all stats
joblib.dump(model_reg, 'random_forest_regressor.pkl')
print("Regression model saved as 'random_forest_regressor.pkl'")

# Save the SVM classification model
joblib.dump(model_clf, 'svm_classifier.pkl')  # Change filename
print("SVM classification model saved as 'svm_classifier.pkl'")

# Save the scaler all stats (used to preprocess the features)
joblib.dump(scaler, 'scaler.pkl')
print("Scaler saved as 'scaler.pkl'")

# Save the regression model(points only)
joblib.dump(model_reg_pts_only, 'random_forest_regressor_pts_only.pkl')
print("regression model of points only is saved")

# Save the scaler(points only) (used to preprocess the features)
joblib.dump(scaler_pts_only, 'scaler_pts_only.pkl')
print("scaler of points only is saved")
