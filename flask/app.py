from flask import Flask, render_template, request
import joblib, os

app = Flask(__name__)
model_path = os.path.join(os.path.dirname(__file__), "../model/model_joblib_diabetes")

def predict_diabetes():
    values = [float(entry.get()) for entry in entries]

    model = joblib.load(model_path)
    prediction = model.predict([values])

    predicted = "Non-Diabetic" if prediction == 0 else "Diabetic"
    result_label.config(text=predicted)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract form data
    data = request.form
    pregnancies = float(data['pregnancies'])
    glucose = float(data['glucose'])
    blood_pressure = float(data['blood_pressure'])
    skin_thickness = float(data['skin_thickness'])
    insulin = float(data['insulin'])
    bmi = float(data['bmi'])
    diabetes_pedigree_function = float(data['diabetes_pedigree_function'])
    age = int(data['age'])

    prediction = (glucose > 120)

    result = 'Diabetic' if prediction else 'Not Diabetic'
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
