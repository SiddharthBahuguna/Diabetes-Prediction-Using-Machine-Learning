import streamlit as st
import joblib
import os

# Get dir path of current file and load the existing model using joblib

model_path = os.path.join(os.path.dirname(__file__), "../../model/model_joblib_diabetes")
model = joblib.load(model_path)

# Page Layout

st.markdown("""
## Diabetes Prediction using Machine Learning
##### This page uses a pre-built model
Slide the bars to adjust values and click on the **Predict** button to check if you are diabetic or not.

Hover over **?** for more information on the data input
""")
st.text("Have a look at the side panel for the model information\n", help="STATS FOR NERDS")
st.write("---")

# -------------------------------------------------------------------------------------------------------------- #


# Sliding bar for every feature input

Pregnancies = st.slider(":red[**Pregnancies**]", 0, 20, 0, help="Number of times pregnant")
st.write("---")

Glucose = st.slider(":red[**Glucose**]",40, 250, 125, help="Plasma glucose concentration a 2 hours in an oral glucose tolerance test")
st.write("---")

Blood_Pressure = st.slider(":red[**Blood Pressure $(mm.Hg)$**]", 30, 160, 90, help="Diastolic blood pressure (mm Hg)")
st.write("---")

Skin_Thickness = st.slider(":red[**Skin_Thickness $(mm)$**]", 5, 100, 20, help="Triceps skin fold thickness (mm)")
st.write("---")

Insulin = st.slider(":red[**Insulin $(mu U/ml)$**]", 10, 500, 80,help="2-Hour serum insulin (mu U/ml)")
st.write("---")

Bmi = st.slider(":red[**BMI $(kg/m^{2})$**]", 10, 80, 22, help="Body mass index (weight in kg/(height in m)^2)")
st.write("---")

Diabetes_Ped_Func = st.slider(":red[**Diabetes Pedigree Function**]", 0.08, 2.42, 0.5, 0.01, 
                                help="Function that scores the probability of diabetes based on family history")
st.write("---")

Age = st.slider(":red[**Age**]", 18, 90, 30, help="Age in years")
st.write("---")

# Show prediction on button click

features = [[Pregnancies, Glucose, Blood_Pressure, Skin_Thickness, Insulin, Bmi, Diabetes_Ped_Func, Age]]
if(st.button(":white[**Predict**]", type="primary")):
    prediction = model.predict(features)
    if(prediction == [1]):
        st.write("**The person is diabetic**")
    else:
        st.write("**The person is *not* diabetic**")

# -------------------------------------------------------------------------------------------------------------- #


# Sidebar layout

# Function to get features sorted according to importance

def get_important_features(model):
    # get feature names from saved model
    feature_names = model.feature_names_in_
    # converting feature names and feature importance to a dictionary
    important_features_dict = dict(zip(feature_names, model.feature_importances_.tolist()))
    # sorting feature names according to the importance
    important_features_sorted = dict(sorted(important_features_dict.items(), key= lambda item: item[1] , reverse=True))
    
    important_features = ""
    num = 1
    # unpacking and returning in a string format
    for (feature, importance) in important_features_sorted.items():
        important_features += f"{num}. {feature} = {format(importance*100, ".2f")}%\n"
        num+=1
    return important_features

# Function to get model parameter values

def get_model_params(model):
    model_params = ""
    num = 1
    for (param, value) in model.get_params().items():
        model_params += f"{num}. {param}= {value}\n"
        num+=1
    return model_params


# Display Model Information

st.sidebar.header("Model Information")

# Model Name

with st.sidebar.expander("Model Name"):
    st.text(f"{type(model).__name__}")
st.sidebar.divider()

# Model Parameters

model_params_expander = st.sidebar.expander("Model Parameters")
model_params = get_model_params(model)
model_params_expander.markdown(model_params)
st.sidebar.divider()

# Display factors that contribute to diabetes (most to least)

important_features_expander = st.sidebar.expander("% Contribution towards diabetes")
important_features = get_important_features(model)
important_features_expander.markdown(important_features)
st.sidebar.divider()

# Training Loss in first and last iterations

with st.sidebar.expander("Loss Calculated in"):
    st.text(f"First iteration: {format(model.train_score_[0],'.3f')}")
    st.text(f"Last iteration: {format(model.train_score_[-1],'.3f')}")
st.sidebar.divider()
