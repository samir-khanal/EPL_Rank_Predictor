# ⚽ EPL Rank Predictor: Machine Learning-Powered Football Analytics

Predict English Premier League team rankings using historical season data with dual ML approaches - **regression** for exact positions and **classification** for category prediction (Top 4, Mid-Table, Relegation Zone). Powered by Flask API + Streamlit UI.

**GitHub Code**: [EPL Rank Predictor Repository](https://github.com/samir-khanal/EPL_Rank_Predictor)

---

## 🧠 Key Features
- **Dual Prediction System**  
  📈 **Regression**: Random Forest model for exact league position (1-20)  
  🏷️ **Classification**: SVM model for category prediction (Top 4/Europe/Mid-Table/Relegation)
  
- **Full-Stack Implementation**  
  🔙 **Backend**: Flask API with model serving  
  🖥️ **Frontend**: Streamlit interactive interface
  
- **Data Insights**  
  📊 Historical analysis of 10+ EPL seasons  
  🔍 Key performance metrics: Wins, Goals, Shots, Points

---

## 🛠️ Tech Stack
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/ScikitLearn-1.6.1-orange)
![Flask](https://img.shields.io/badge/Flask-3.0.0-lightgrey)
![Streamlit](https://img.shields.io/badge/Streamlit-1.36.0-%23FF4B4B)

```bash
# Core Dependencies
pandas==2.2.3
numpy==1.26.4
scikit-learn==1.6.1
Flask==3.0.0
streamlit==1.36.0
```

## 🚀 Getting Started
### Installation
```
git clone https://github.com/samir-khanal/EPL_Rank_Predictor.git
cd EPL_Rank_Predictor

python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

pip install -r requirements.txt
```
## Start Flask API:
```python server.py
```
API will run at http://localhost:5000
### Launch Streamlit Frontend:
```
streamlit run streamlit_app.py
```
Access UI at http://localhost:8501

## 🤝Contributing
PRs welcome! See CONTRIBUTING.md for guidelines.
