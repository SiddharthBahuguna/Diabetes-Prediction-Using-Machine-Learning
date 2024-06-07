
## Diabetes Prediction Using Machine Learning

This repository contains a machine learning project to predict diabetes using the PIMA Indians Diabetes Database. The project involves data preprocessing, model training, evaluation, and a GUI for predictions.
## Project Structure

Diabetes Prediction.ipynb: Jupyter Notebook with code for data preprocessing, model training, and evaluation.

diabetes.csv: Dataset used for training the model.

model_joblib_diabetes: Serialized model file for making predictions.

gui.py: Python script for the graphical user interface (GUI).
## Requirements
Python 3.x

Jupyter Notebook

pandas

numpy

scikit-learn

joblib

tkinter
## Installation

1. Clone the repository:

```bash
  git clone https://github.com/SiddharthBahuguna/Diabetes-Prediction-Using-Machine-Learning.git

```

2. Install the required packages:

```bash
pip install pandas numpy scikit-learn joblib

```


## Usage
## Running the Jupyter Notebook

1. Open the Jupyter Notebook:

```bash
jupyter notebook Diabetes\ Prediction.ipynb

```

2. Run the cells to preprocess the data, train the model, and evaluate its performance.

3. Use the saved model (model_joblib_diabetes) for predictions on new data.




## Running the GUI

1. Ensure you have tkinter installed. For most Python installations, it comes pre-installed. If not, you can install it using your system's package manager.

2. Run the gui.py script:
```bash
python gui.py

```

3. Enter the required input values and click on the "Predict" button to get the prediction result.

4. Press Shift+Enter after last line of code(i.e mainloop()) for GUI.


## GUI Code

```bash
from tkinter import *
import joblib
import numpy as np

def show_entry_fields():
    p1 = float(e1.get())
    p2 = float(e2.get())
    p3 = float(e3.get())
    p4 = float(e4.get())
    p5 = float(e5.get())
    p6 = float(e6.get())
    p7 = float(e7.get())
    p8 = float(e8.get())

    model = joblib.load('model_joblib_diabetes')
    result = model.predict([[p1, p2, p3, p4, p5, p6, p7, p8]])

    if result == 0:
        Label(master, text="Non-Diabetic").grid(row=31)
    else:
        Label(master, text="Diabetic").grid(row=31)

master = Tk()
master.title("Diabetes Prediction Using Machine Learning")

label = Label(master, text="Diabetes Prediction Using Machine Learning", bg="black", fg="white").grid(row=0, columnspan=2)

Label(master, text="Pregnancies").grid(row=1)
Label(master, text="Glucose").grid(row=2)
Label(master, text="BloodPressure").grid(row=3)
Label(master, text="SkinThickness").grid(row=4)
Label(master, text="Insulin").grid(row=5)
Label(master, text="BMI").grid(row=6)
Label(master, text="DiabetesPedigreeFunction").grid(row=7)
Label(master, text="Age").grid(row=8)

e1 = Entry(master)
e2 = Entry(master)
e3 = Entry(master)
e4 = Entry(master)
e5 = Entry(master)
e6 = Entry(master)
e7 = Entry(master)
e8 = Entry(master)

e1.grid(row=1, column=1)
e2.grid(row=2, column=1)
e3.grid(row=3, column=1)
e4.grid(row=4, column=1)
e5.grid(row=5, column=1)
e6.grid(row=6, column=1)
e7.grid(row=7, column=1)
e8.grid(row=8, column=1)

Button(master, text='Predict', command=show_entry_fields).grid()

mainloop()


```
## Screenshots

![nd](https://github.com/SiddharthBahuguna/Diabetes-Prediction-Using-Machine-Learning/assets/112819453/58a43b40-76c4-471f-b143-bc5d619e3648)



## Acknowledgements

PIMA Indians Diabetes Database

## License

[MIT](https://choosealicense.com/licenses/mit/)

