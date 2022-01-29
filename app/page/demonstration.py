import streamlit as st
import requests

URL_PREPROCESS_VIDEO = "mon_url"

def app():
    st.title("Video upload !")        
            
    video_file = st.file_uploader("Upload the video", type=["mp4"], key="files")
    
    if video_file is not None:
        files = {"file" : video_file}
        json_response = requests.post(url=URL_PREPROCESS_VIDEO, files=files)
        
        st.video(video_file.read())
    

    