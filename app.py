from flask import Flask, request,render_template
import joblib
import numpy as np

model=joblib.load('model/heart_risk_prediction_regression_model.sav')

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/getresults', methods=['POST'])
def getrusults():

    results = request.form
    name= results['name']
    gender = float(results['gender'])
    age = float(results['age'])
    tc= float(results['tc'])
    hdl = float(results['hdl'])
    sbp = float(results['sbp'])
    smoke= float(results['smoke'])
    bpm = float(results['bpm'])
    diabetes = float(results['diab'])

    test_data =np.array([gender,age,tc,hdl,smoke,bpm,diabetes]).reshape(1, -1)
    prediction = model.predict(test_data)
    resultDict={"name": name, "risk": round(prediction[0][0],2)}

    return render_template('result.html', results=resultDict)
if __name__ == '__main__':
    app.run(debug=True)





