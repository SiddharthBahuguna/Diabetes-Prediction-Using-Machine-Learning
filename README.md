
## Diabetes Prediction Using Machine Learning

This repository contains a machine learning project to predict diabetes using the PIMA Indians Diabetes Database. The project involves data preprocessing, model training, evaluation, and a GUI for predictions.
## Project Structure

Diabetes Prediction.ipynb: Jupyter Notebook with code for data preprocessing, model training, and evaluation.

diabetes.csv: Dataset used for training the model.

model_joblib_diabetes: Serialized model file for making predictions.

gui/gui.py: Python script for the graphical user interface (GUI).
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
pip install pandas numpy scikit-learn joblib tkinter

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
python gui/gui.py
```

3. Enter the required input values and click on the "Predict" button to get the prediction result.

## Screenshots

![nd](https://github.com/SiddharthBahuguna/Diabetes-Prediction-Using-Machine-Learning/assets/112819453/58a43b40-76c4-471f-b143-bc5d619e3648)



## Acknowledgements

PIMA Indians Diabetes Database

## License

[MIT](LICENSE)

