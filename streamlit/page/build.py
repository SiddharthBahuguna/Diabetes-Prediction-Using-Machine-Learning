import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time
import base64
import os
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV

# HELP CONSTANTS

PREPROCESS_HELP = "Clean the data (Data imputation, Outlier Removal, Feature Engineering, Scaling, One Hot Encoding) If checked, will increase the time to build model"
N_ESTIMATORS_HELP = "The number of trees in the forest."
CRITERION_HELP = "The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “log_loss” and “entropy” both for the Shannon information gain"
MAX_DEPTH_HELP = "The maximum depth of the tree."
MIN_SAMPLES_SPLIT_HELP = "The minimum number of samples required to split an internal node."
MIN_SAMPLES_LEAF_HELP = "The minimum number of samples required to be at a leaf node."
MAX_FEATURES_HELP = "The number of features to consider when looking for the best split."
BOOTSRAP_HELP = "Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree."
OOB_SCORE_HELP = "Whether to use out-of-bag samples to estimate the generalization score. Only available if bootstrap=True."
N_JOBS_HELP = "The number of jobs to run in parallel."
RANDOM_STATE_HELP = "Controls both the randomness of the bootstrapping of the samples used when building trees (if bootstrap=True) and the sampling of the features to consider when looking for the best split at each node (if max_features < n_features)."

# -------------------------------------------------------------------------------------------------------------- #

# Sidebar layout

# Options to choose input parameter values

preprocess = st.sidebar.checkbox("Apply Data Pre-Processing", value=True, help=PREPROCESS_HELP)

st.sidebar.header("Set Parameters")
st.sidebar.warning("Parameter values set at random may result in long building time")
split_size = st.sidebar.slider("Data split ratio (% for training)", 10, 90, 80, 5)

st.sidebar.subheader("Learning Parameters")
parameter_n_estimators = st.sidebar.slider("Number of estimators (n_estimators)", 1, 1000, (10,50), 50, help=N_ESTIMATORS_HELP)
parameter_n_estimators_step = st.sidebar.number_input("Step size for n_estimators", 10, 100, step=5)
st.sidebar.write("---")

parameter_max_features = st.sidebar.slider("Max features (max_features)", 1, 10, (1,3), 1, help=MAX_FEATURES_HELP)
st.sidebar.write("---")
parameter_max_depth = st.sidebar.slider("Max depth (max_depth)", 1, 10, (3,6), 1, help=MAX_DEPTH_HELP)
st.sidebar.write("---")

parameter_min_samples_split = st.sidebar.slider("Minimum number of samples required to split an internal node (min_samples_split)", 2, 10, (2,3), 1, help=MIN_SAMPLES_SPLIT_HELP)
parameter_min_samples_leaf = st.sidebar.slider("Minimum number of samples required to be at leaf node (min_samples_leaf)", 1, 10, 2, 1, help=MIN_SAMPLES_LEAF_HELP)
st.sidebar.write("---")

st.sidebar.subheader("General Parameters")
parameter_random_state = st.sidebar.number_input('Seed number (random_state)', 0, 12345, 42, step=1, help=RANDOM_STATE_HELP)
st.sidebar.write("---")
parameter_criterion = st.sidebar.select_slider('Performance measure (criterion)', options=['gini', 'entropy', 'log_loss'], help=CRITERION_HELP)
st.sidebar.write("---")
parameter_bootstrap = st.sidebar.select_slider('Bootstrap samples when building trees (bootstrap)', options=[True, False], help=BOOTSRAP_HELP)
st.sidebar.write("---")
parameter_oob_score = st.sidebar.select_slider('Whether to use out-of-bag samples to estimate the generalization score. (oob_score)', options=[False, True], help=OOB_SCORE_HELP)
st.sidebar.write("---")
parameter_n_jobs = st.sidebar.select_slider('Number of jobs to run in parallel (n_jobs)', options=[1, -1], help=N_JOBS_HELP)
st.sidebar.write("---")

# Create range variables for appropriate features

n_estimators_range = np.arange(parameter_n_estimators[0], parameter_n_estimators[1]+parameter_n_estimators_step, parameter_n_estimators_step)
max_features_range = np.arange(parameter_max_features[0], parameter_max_features[1]+1, 1)
min_samples_split_range = np.arange(parameter_min_samples_split[0], parameter_min_samples_split[1]+1, 1)
max_depth_range = np.arange(parameter_max_depth[0], parameter_max_depth[1]+1, 1)
param_grid = dict(max_features=max_features_range, n_estimators=n_estimators_range, min_samples_split=min_samples_split_range, max_depth=max_depth_range)


# -------------------------------------------------------------------------------------------------------------- #


# Utility Functions

def download_model_performance(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="model_performance.csv">Download Model Performance</a>'
    return href

def preprocess_data(df):
    
    df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.nan)
    
    columns = df.columns
    columns = columns.drop("Outcome")
    
    # Filling incomplete values based on their outcome class' median value
    def median_target(var):   
        temp = df[df[var].notnull()]
        temp = temp[[var, 'Outcome']].groupby(['Outcome'])[[var]].median().reset_index()
        return temp
    
    for i in columns:
        median_target(i)
        df.loc[(df['Outcome'] == 0 ) & (df[i].isnull()), i] = median_target(i)[i][0]
        df.loc[(df['Outcome'] == 1 ) & (df[i].isnull()), i] = median_target(i)[i][1]
    
    # ----------------- #

    # Remove Outliers using InterQuartile method
    Q1 = df.Insulin.quantile(0.25)
    Q3 = df.Insulin.quantile(0.75)
    IQR = Q3-Q1
    lower = Q1 - 1.5*IQR
    upper = Q3 + 1.5*IQR
    df.loc[df["Insulin"] > upper,"Insulin"] = upper
    
    lof = LocalOutlierFactor(n_neighbors= 10)
    lof.fit_predict(df)
    df_scores = lof.negative_outlier_factor_
    threshold = np.sort(df_scores)[7]
    
    outlier = df_scores > threshold
    df = df[outlier]

    # ----------------- #

    # Feature Engineering (Binning to create new features with discrete values for better identification)

    # On Insulin
    NewBMI = pd.Series(["Underweight", "Normal", "Overweight", "Obesity 1", "Obesity 2", "Obesity 3"], dtype = "category")
    df["NewBMI"] = NewBMI
    df.loc[df["BMI"] < 18.5, "NewBMI"] = NewBMI[0]
    df.loc[(df["BMI"] > 18.5) & (df["BMI"] <= 24.9), "NewBMI"] = NewBMI[1]
    df.loc[(df["BMI"] > 24.9) & (df["BMI"] <= 29.9), "NewBMI"] = NewBMI[2]
    df.loc[(df["BMI"] > 29.9) & (df["BMI"] <= 34.9), "NewBMI"] = NewBMI[3]
    df.loc[(df["BMI"] > 34.9) & (df["BMI"] <= 39.9), "NewBMI"] = NewBMI[4]
    df.loc[df["BMI"] > 39.9 ,"NewBMI"] = NewBMI[5]

    def set_insulin(row):
        if row["Insulin"] >= 16 and row["Insulin"] <= 166:
            return "Normal"
        else:
            return "Abnormal"
    
    df = df.assign(NewInsulinScore=df.apply(set_insulin, axis=1))

    # On Glucose (Binning)
    NewGlucose = pd.Series(["Low", "Normal", "Overweight", "Secret", "High"], dtype = "category")
    df["NewGlucose"] = NewGlucose
    df.loc[df["Glucose"] <= 70, "NewGlucose"] = NewGlucose[0]
    df.loc[(df["Glucose"] > 70) & (df["Glucose"] <= 99), "NewGlucose"] = NewGlucose[1]
    df.loc[(df["Glucose"] > 99) & (df["Glucose"] <= 126), "NewGlucose"] = NewGlucose[2]
    df.loc[df["Glucose"] > 126 ,"NewGlucose"] = NewGlucose[3]

    # ----------------- #

    # One-hot encoding (Converting the categorical values to numerical by separating them as features)
    df = pd.get_dummies(df, columns =["NewBMI","NewInsulinScore", "NewGlucose"], drop_first = True)
    categorical_df = df[['NewBMI_Obesity 1','NewBMI_Obesity 2', 'NewBMI_Obesity 3', 'NewBMI_Overweight','NewBMI_Underweight',
                        'NewInsulinScore_Normal','NewGlucose_Low','NewGlucose_Normal', 'NewGlucose_Overweight', 'NewGlucose_Secret']]

    # Separating target variable and feature_vector

    y = df["Outcome"]
    X = df.drop(["Outcome",'NewBMI_Obesity 1','NewBMI_Obesity 2', 'NewBMI_Obesity 3', 'NewBMI_Overweight','NewBMI_Underweight',
                        'NewInsulinScore_Normal','NewGlucose_Low','NewGlucose_Normal', 'NewGlucose_Overweight', 'NewGlucose_Secret'], axis = 1)
    cols = X.columns
    index = X.index

    # ----------------- #

    # Scaling using RobustScaler()
    # Scale features using statistics that are robust to outliers.
    # This Scaler removes the median and scales the data according to
    # the quantile range (defaults to IQR: Interquartile Range). 
    # The IQR is the range between the 1st quartile (25th quantile) 
    # and the 3rd quartile (75th quantile).

    transformer = RobustScaler().fit(X)
    X = transformer.transform(X)
    X = pd.DataFrame(X, columns = cols, index = index)

    # Join the scaled data (numerical) and one-hot encoded data
    X = pd.concat([X,categorical_df], axis = 1)
    processed_df = pd.concat([X,y], axis=1 )
    return processed_df


def build_model(df):

    progress_bar.progress(20, "Building Model, don't run away!")
    time.sleep(0.5)
    
    # dividing the datafram into features and target variables
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]

    # Data Splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_size, random_state=parameter_random_state)
    
    # Quote by the Data Professor
    progress_bar.progress(30, "The best way to learn Data Science, is to do Data Science - Data Professor (This might take a whille)")

    # Applying Parameters to the RFC model and fitting using GridSearchCV
    rfc = RandomForestClassifier(
        n_estimators=parameter_n_estimators,
        max_depth=parameter_max_depth,
        max_features=parameter_max_features,
        criterion=parameter_criterion,
        min_samples_split=parameter_min_samples_split,
        min_samples_leaf=parameter_min_samples_leaf,
        bootstrap=parameter_bootstrap,
        oob_score=parameter_oob_score,
        n_jobs=parameter_n_jobs
    )

    grid = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=10, scoring="accuracy", verbose=2)
    grid.fit(X_train, y_train)

    progress_bar.progress(80, "Testing Model, Almost there!")
    time.sleep(0.5)

    # Predict from the test set
    y_pred_train = grid.predict(X_train)
    y_pred_test = grid.predict(X_test)

    # ----------------- #

    # Display performance metrics
    def get_metrics(y, y_pred):
        acc = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        roc_auc = roc_auc_score(y, y_pred)
        loss = log_loss(y, y_pred)
        return [acc, precision, recall, f1, roc_auc, loss]

    st.subheader("Model Performance on Training and Testing sets")
    metrics_index = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC', 'Log Loss']
    metrics = {'Train': get_metrics(y_train, y_pred_train), 'Test': get_metrics(y_test, y_pred_test)}
    metrics_table = pd.DataFrame(metrics, index=metrics_index)
    st.write(metrics_table)

    st.info("The best parameters are **%s** with a score of **%0.2f**"% (grid.best_params_, grid.best_score_))

    st.subheader('Model Parameters')
    st.write(grid.get_params())

    progress_bar.progress(95, "Plotting Charts using Plotly, don't blink!")
    time.sleep(0.5)

    # Get significance score of variables
    rfc_tuned = RandomForestClassifier(**grid.best_params_)
    rfc_tuned.fit(X, y)
    feature_imp = pd.Series(rfc_tuned.feature_importances_, index=X.columns).sort_values(ascending=False)
    significance_df = pd.DataFrame({'Significance Score Of Variables': feature_imp, 'Variables': feature_imp.index})
    fig = px.bar(significance_df, x='Significance Score Of Variables', y='Variables', title='Variable Severity Levels of the Best Model')
    st.plotly_chart(fig)

    # ----------------- #

    # Process grid data

    grid_results = pd.DataFrame(grid.cv_results_)[['param_max_features','param_n_estimators','mean_test_score']]

    # Segment data into groups based on the 2 hyperparameters

    grid_contour = grid_results.groupby(['param_max_features','param_n_estimators']).mean()

    # Pivoting the data

    grid_reset = grid_contour.reset_index()
    grid_reset.columns = ['param_max_features', 'param_n_estimators', 'mean_test_score']
    grid_pivot = grid_reset.pivot(index = 'param_max_features', columns = 'param_n_estimators')
    
    x = grid_pivot.columns.levels[1].values
    y = grid_pivot.index.values
    z = grid_pivot.values

    # ----------------- #
    # Plotting data

    fig = go.Figure(data=[go.Surface(x=x, y=y, z=z, colorscale='viridis')])
    fig.update_layout(title='Hyperparameter tuning',
                    scene=dict(
                        xaxis_title='n_estimators',
                        yaxis_title='max_features',
                        zaxis_title='Accuracy_score'),
                    autosize=False,
                    width=800, height=800,
                    margin=dict(l=65, r=50, b=65, t=90))
    st.plotly_chart(fig)

    #-----Save grid data-----#
    x = pd.DataFrame(x)
    y = pd.DataFrame(y)
    z = pd.DataFrame(z)
    df = pd.concat([x,y,z], axis=1)
    st.markdown(download_model_performance(grid_results), unsafe_allow_html=True)

    progress_bar.progress(100, "All done!")


# -------------------------------------------------------------------------------------------------------------- #

# Page Layout

st.markdown(
"""
### Build your own Random Forest Classifier by Hypertuning Parameters!
To learn more on how an RFC works, 
check out [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).
""")


filepath = os.path.join(os.path.dirname(__file__), "../../data/diabetes.csv")
df = pd.read_csv(filepath)
st.markdown("#### Example Data")
st.write(df.head())

# Link to download the dataset

st.markdown("""[Dataset Source](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)""")
st.text("Click on the button below to build your model after setting the parameters. \nTo avoid unknown behaviour, do not press the button until current model is completely built")

# Set progress bar to 0 initially, and update as and when the model builds

progress_bar = st.progress(0)

# Start building the model, on button's click

if(st.button("Build Model", type='primary')):
    st.session_state.predict_button = True
    progress_bar.progress(0, "Pre-processing Data, sit back & relax")
    time.sleep(0.5)

    # If data preprocessing is to be applied
    if(preprocess):
        processed_df = preprocess_data(df)
        build_model(processed_df)
    else:
        build_model(df)