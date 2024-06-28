import streamlit as st
import pandas as pd
from ydata_profiling import ProfileReport
# from streamlit_pandas_profiling import st_profile_report
import os

def visualize_page():

    # Get path of the dataset
    filepath = os.path.join(os.path.dirname(__file__), "../../data/diabetes.csv")
    
    st.title("Exploratory Data Analysis")
    st.markdown(
        """
        **ydata-profiling** (previously, Pandas Profiling) is used to generate this extensive report.
        """
    )
    
    with st.sidebar.header("Upload your CSV Dataset for an EDA!"):
        uploaded_file = st.sidebar.file_uploader("Input CSV file here", type=['csv'])
    
    # If no file uploaded
    if uploaded_file is not None:
        
        df = pd.read_csv(uploaded_file)
        profile_report = ProfileReport(df, explorative=True, dark_mode=True)
        st.subheader("**Profiling Report will be generated in your local directory**")
        st.subheader("Input DataFrame")
        st.write(df.head(5))

        st.divider()
        st.subheader("**Please wait while the file is being rendered**")
        
        # Open EDA.html and save in current working directory
        profile_report.to_file("custom_EDA.html", silent=False)
        st.text("custom_EDA.html file has been saved in your current working directory.")

    # if no file uploaded for EDA
    else:
        st.info("Upload a CSV Dataset or use Default Dataset")
        
        # If default dataset is to be used
        if st.button("Press to use PIMA Indians Diabetes Dataset"):
            @st.cache_data
            def load_data():
                return pd.read_csv(filepath)
            df = load_data()

            profile_report = ProfileReport(df, explorative=True, dark_mode=True)
            st.subheader("**Profiling Report will be generated in your local directory**")
            st.subheader("Input DataFrame")
            st.write(df.head(5))

            st.divider()

            st.subheader("**Please wait while the file is being rendered**")
            # Open EDA.html and save in current working directory
            profile_report.to_file("EDA.html", silent = False)
            st.text("EDA.html file has been saved in your current working directory.")

visualize_page()