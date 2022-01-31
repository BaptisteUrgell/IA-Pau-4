import streamlit as st

st.set_page_config(
     page_title="CY Riders DEMO",
     page_icon="🍃",
     #layout="wide",
     menu_items={
         'Get Help': 'https://www.extremelycoolapp.com/help',
         'Report a bug': "https://www.extremelycoolapp.com/bug",
         'About': "# This is a header. This is an *extremely* cool app!"
     }
 )

from app.config.config import get_api_settings
import streamlit as st
import cv2
from surfnet_v2.src.tracking.track_video import Display
import os
from surfnet_v2.src.track import main
import time
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np

settings = get_api_settings()

LOGO_SURFRIDERS = settings.logo_surfriders
LOGO_IA_PAU = settings.logo_ia_pau


# app = MultiPage()

# Title of the main page
st.title("IA PAU 4 - CY Riders")


hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            #MainMenu {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

c0, c1 = st.columns([0.6,0.4])

with c0:
    st.image(LOGO_SURFRIDERS)
with c1:
    st.image(LOGO_IA_PAU)
    
    
    
st.markdown('---')
        
st.title("Test Model on Custom Video")        
        
        
# video = open(os.path.join('/home/eisti/Perso/Projets/ia-pau-4/IA-Pau-4/app/tmp/output/', 'testvid.webm'), 'rb')
# video_bytes = video.read()
# st.video(video_bytes, format="video/mp4")


        
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
        video = open(os.path.join('/home/eisti/Perso/Projets/ia-pau-4/IA-Pau-4/app/tmp/output/', video_file.name.split('.')[0] + '.webm'), 'rb')

        video_bytes = video.read()
        st.video(video_bytes)
        
        st.subheader('Detection report :')              
        
        df = pd.read_csv("/home/eisti/Perso/Projets/ia-pau-4/IA-Pau-4/app/tmp/data/current_detection_analysis.csv")
        
        df = df.T.astype(str)
        df_tmp = df.iloc[1: , :]
        df_tmp.columns = ["Count"]
        st.dataframe(df_tmp)
				    
        # labels = np.array(df.columns)[1:]
        # #print(labels, len(labels))

        #sizes = np.array(df.values)[0][1:]
        # #print(sizes, len(sizes))
        # fig = plt.figure()
        # ax = fig.add_axes([0,0,1,1])
        # ax.bar(labels, sizes)
        # plt.xticks(rotation=45)
        # st.pyplot(fig)
                
        
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

