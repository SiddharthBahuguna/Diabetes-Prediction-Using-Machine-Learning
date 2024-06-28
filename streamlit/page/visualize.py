import streamlit as st
import pandas as pd
from pandas_profiling import ProfileReport
# from streamlit_pandas_profiling import st_profile_report
import os

def visualize_page():

    filepath = os.path.join(os.path.dirname(__file__), "../../data/diabetes.csv")
    st.title("Exploratory Data Analysis")
    st.markdown(
        """
        **Pandas Profiling** is used to generate this extensive report.
        """
    )
    
    with st.sidebar.header("Upload your CSV Dataset for an EDA!"):
        uploaded_file = st.sidebar.file_uploader("Input CSV file here", type=['csv'])
    
    if uploaded_file is not None:
        
        # @st.cache_data
        def load_csv():
            csv = pd.read_csv(uploaded_file)
            return csv
        
        df = load_csv()
        profile_report = ProfileReport(df, explorative=True)
        st.subheader("**Pandas Profiling Report will be generated in your local directory**")
        st.subheader("Input DataFrame")
        st.write(df.head(5))

        st.divider()

        st.subheader("**Please wait while the file is being rendered**")
        st.text("You can download using the button below:")
        # Save in local directory
        profile_report.to_file("profile_report.html")
        # Option to download the file
        st.download_button("Click to download Profiling Report", "html", "profile_report.html", type='primary')
        
    else:
        st.info("Upload a CSV Dataset or use Default Dataset")
        
        if st.button("Press to use PIMA Indians Diabetes Dataset"):
            @st.cache_data
            def load_data():
                return pd.read_csv(filepath)
            df = load_data()
        
            profile_report = ProfileReport(df, explorative=True)
            st.subheader("**Pandas Profiling Report will be generated in your local directory**")
            st.subheader("Input DataFrame")
            st.write(df.head(5))

            st.divider()

            st.subheader("**Please wait while the file is being rendered**")
            st.text("Alternatively, you can download using the button below:")
            # Save in local directory
            profile_report.to_file("profile_report.html")
            # Option to download the file
            st.download_button("Click to download Profiling Report", "html", "profile_report.html", type='primary')

visualize_page()