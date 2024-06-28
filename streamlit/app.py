import streamlit as st

st.set_page_config(
    page_title="Diabetes Prediction Application using ML", layout="wide", initial_sidebar_state="expanded",
    menu_items= {
            "Get help": "https://github.com/SiddharthBahuguna/Diabetes-Prediction-Using-Machine-Learning/blob/main/README.md",
            "Report a Bug": "https://github.com/SiddharthBahuguna/Diabetes-Prediction-Using-Machine-Learning/issues",
            "About": 
            """
            Contributed with :heart: by [Karthik Rao](https://github.com/raokarthik15)
            
            Credits:
            - [Data Professor](https://www.youtube.com/@DataProfessor)
            - [Dataset - UCI ML](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
            - [Ahmet Can Karaoglan](https://www.kaggle.com/code/ahmetcankaraolan/diabetes-prediction-using-machine-learning#1-Exploratory-Data-Analysis)
            - [scikit-learn](https://scikit-learn.org/stable/)
            - [Plotly](https://plotly.com/)
            - [Streamlit](https://docs.streamlit.io/develop/api-reference)
            """
        })

# Add pages to navigate on the sidebar

pg = st.navigation(
    
    [st.Page("page/predict.py", title="Predict", icon=":material/conditions:"), 
    st.Page("page/build.py", title="Build Model", icon=":material/build:"),
    st.Page("page/visualize.py", title="Visualize Data", icon=":material/eda:")], 
    
    position="sidebar")

pg.run()