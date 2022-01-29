import streamlit as st
import requests
import copy
import cv2
from surfnet_v2.src.tracking.track_video import Display
import os
from surfnet_v2.src.track import main
from ..config.config import get_api_settings
import time
import pandas as pd
import json

settings = get_api_settings()

APP_DIR = settings.root_dir

# URL_PREPROCESS_VIDEO = "http://127.0.0.1:5000/"

def app():
    st.title("Upload your Video")        
            
    video_file = st.file_uploader("Upload the video", type=["mp4", "avi"], key="files")
         
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
        container_empty.empty()
        with st.spinner(f"Video Building..."):
            time.sleep(5)
            video = open(os.path.join('/home/eisti/Perso/Projets/ia-pau-4/IA-Pau-4/app/tmp/output', video_file.name.split('.')[0] + '.avi'), 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)
                              
            df = pd.read_csv("/home/eisti/Perso/Projets/ia-pau-4/IA-Pau-4/app/tmp/data/current_detection_analysis.csv")
            st.bar_chart(df.T)
            
            with open("/home/eisti/Perso/Projets/ia-pau-4/IA-Pau-4/app/tmp/data/consumption_report.json", 'r') as f:
	            data = json.load(f)
        
            st.json(data)



