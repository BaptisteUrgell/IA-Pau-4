import streamlit as st
import requests
import copy
import cv2
from surfnet_v2.src.tracking.track_video import Display
import os
from surfnet_v2.src.track import main
from ..config.config import get_api_settings

settings = get_api_settings()

APP_DIR = settings.root_dir

# URL_PREPROCESS_VIDEO = "http://127.0.0.1:5000/"

def app():
    st.title("Upload your Video")        
            
    video_file = st.file_uploader("Upload the video", type=["mp4"], key="files")
         
    if video_file is not None:
        
        with st.spinner(f"The Model is processing video : {video_file.name} ..."):
            
            # video_tmp = copy.copy(video_file)
            with open(os.path.join("app", "tmp", video_file.name),"wb") as f:
                f.write(video_file.getbuffer())
                
            display = Display(on=False, interactive=True)
            container_empty = st.empty()
            video = cv2.VideoCapture(os.path.join("app", "tmp", video_file.name))
            main(None, display, video_raw=video, demo=True, demo_container=container_empty, video_name=video_file.name) # MAJ Temps réel des predictions
            
            # video_with_overlay = f2(json_inferences)

            #files = {"file" : video_file}
            #json_response = requests.post(url=URL_PREPROCESS_VIDEO, files=files)
        
        with open(os.path.join("app", "tmp", video_file.name),"wb") as f:
                f.write(video_file.getbuffer())
        # video_bytes = video_with_overlay.read()
        # st.video(video_bytes)

