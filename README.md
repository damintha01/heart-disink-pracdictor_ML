# Heart Disease Risk Predictor

A web application that predicts the risk level of heart disease based on various health parameters. This application uses machine learning to provide risk assessments and helps users understand their potential heart health risks.

## 🚀 Features

- User-friendly web interface
- Real-time risk prediction
- Responsive design
- Professional medical-themed UI
- Instant results visualization

## 📊 Parameters Used for Prediction

- Gender
- Age
- Total Cholesterol (TC)
- HDL Cholesterol
- Systolic Blood Pressure (SBP)
- Smoking Status
- Blood Pressure Medication Status
- Diabetes Status

## 🛠️ Technologies Used

- **Backend**: Python, Flask
- **Frontend**: HTML5, CSS3
- **Machine Learning**: scikit-learn, joblib
- **Model**: Regression model for risk prediction

## 💻 Installation

1. Clone the repository:
2. Create and activate a virtual environment:
3. Install required packages:
4. Run the application:
5. Open your browser and navigate to:
```
http://127.0.0.1:5000
```

## 📁 Project Structure

```
website/
│
├── app.py                 # Main Flask application
├── model/
│   └── heart_risk_prediction_regression_model.sav    # Trained ML model
│
├── static/
│   └── css/
│       └── style.css     
│
├── templates/
│   ├── index.html       
│   └── result.html       
│
└── web/                  
```

## 🔧 Usage

1. Fill in the required health parameters in the form:
   - Name
   - Gender (1 for Male, 2 for Female)
   - Age
   - Total Cholesterol (in mg/dL)
   - HDL Cholesterol (in mg/dL)
   - Systolic Blood Pressure (in mm Hg)
   - Smoking Status (1 for Yes, 0 for No)
   - Blood Pressure Medication (1 for Yes, 2 for No)
   - Diabetes Status (1 for Yes, 0 for No)

2. Click "Calculate Risk Level" to see your results
3. View your heart disease risk percentage
4. Use the "Back" button to make another prediction

## ⚠️ Important Notes

- This tool is for educational purposes only
- Always consult healthcare professionals for medical advice
- Results should not be used as a substitute for professional medical diagnosis

## 👥 Authors

- Damintha01 - *Initial work*

## 🙏 Acknowledgments

- Dataset used for training the model
- Contributors and testers
- Open source community

---

⭐️ If you found this project helpful, please give it a star on GitHub!
