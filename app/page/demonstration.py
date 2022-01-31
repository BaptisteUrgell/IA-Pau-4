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

    st.markdown('---')
            
    st.title("Test Model on Custom Video")        
            
            
    video = open(os.path.join('/home/eisti/Perso/Projets/ia-pau-4/IA-Pau-4/app/tmp/output/', 'testvid.avi'), 'rb')
    video_bytes = video.read()
    st.video(video_bytes, format="video/mp4")
    
    video_file = st.file_uploader("Upload your video here", type=["mp4", "avi"], key="files")
         
    if video_file is not None:
        
        with st.spinner(f"The Model is processing video : {video_file.name} ..."):
            
            with open(os.path.join("app", "tmp", video_file.name),"wb") as f:
                f.write(video_file.getbuffer())
                
            display = Display(on=False, interactive=True)
            container_empty = st.empty()
            video = cv2.VideoCapture(os.path.join("app", "tmp", video_file.name))
            main(None, display, video_raw=video, demo=True, demo_container=container_empty, video_name=video_file.name) # MAJ Temps réel des predictions
            

        container_empty.empty()
        with st.spinner(f"Video Building..."):
            # vid_path = os.path.join('/home/eisti/Perso/Projets/ia-pau-4/IA-Pau-4/app/tmp/output/', video_file.name.split('.')[0] + '.avi')
            # vid_mp4_path = vid_path.split('.')[0] + '.mp4'
            # os.system('ffmpeg -i {} -vcodec libx264 {}'.format(vid_path, vid_mp4_path))
            # video = open(os.path.join('/home/eisti/Perso/Projets/ia-pau-4/IA-Pau-4/app/tmp/output/', video_file.name.split('.')[0] + '.avi'), 'rb')
            video = open(os.path.join('/home/eisti/Perso/Projets/ia-pau-4/IA-Pau-4/app/tmp/output/', video_file), 'rb')

            video_bytes = video.read()
            st.video(video_bytes)
            
            st.subheader('Detection report :')              
            
            df = pd.read_csv("/home/eisti/Perso/Projets/ia-pau-4/IA-Pau-4/app/tmp/data/current_detection_analysis.csv")
            st.dataframe(df.astype(str))
            
            with open("/home/eisti/Perso/Projets/ia-pau-4/IA-Pau-4/app/tmp/data/consumption_report.json", 'r') as f:
                data = json.load(f)
            
            
            st.subheader('Consumption report :')
        
            row01, row02, row03 = st.columns(3)
            row02.metric("Inference Duration", str(round(data["duration_seconds"], 3)) + " s")
            row1, row2, row3 = st.columns(3)
            row1.metric("Mean CPU power", str(data["mean_cpu_power_W"]) +  " W", "🍃")
            row2.metric("Mean System power", str(data["mean_system_power_W"]) + " W", "🍃")
            row3.metric("Mean GPU  power", str(data["mean_gpu_power_W"]) + " W", "🍃")

            row11, row22, row33 = st.columns(3)
            row11.metric("Energy CPU consumed", str(data["energy_cpu_consumed_Wh"]) + " Wh", "🍃")
            row22.metric("Energy system consumed", str(data["energy_system_consumed_Wh"]) + " Wh", "🍃")
            row33.metric("Energy GPU  consumed", str(data["energy_gpu_consumed_Wh"]) + " Wh", "🍃")

