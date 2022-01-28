from distutils.command.upload import upload
import streamlit as st
import pandas as pd

def app():
    st.title("Add dataset, image or ")        
            
    files = st.file_uploader("Upload a Dataset", type=["jpeg","csv","json"], key="files", accept_multiple_files=True)
    
    with st.container():
        for file in files:
            if file.type == "text/csv":
                df = pd.read_csv(file)
                st.dataframe(df)
                """
                le code sur le dataframe à faire
                """
            elif file.type == "image/jpeg":
                st.write(file.type)
                st.image(file)
            else:
                st.write(file.type)
    