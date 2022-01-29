import streamlit as st
import requests
import copy
from PIL import Image
import os
from app.config.config import get_api_settings

settings = get_api_settings()

APP_DIR = settings.root_dir
MODEL_INTERPRETATION_IMAGES_DIR = settings.model_interpretations_dir

TEST_IMAGES_DIR = "/home/eisti/Perso/Projets/ia-pau-4/Videos/frames/"

def app():
    st.title("Some Model Informations")        
            
    im_paths = [os.path.join(MODEL_INTERPRETATION_IMAGES_DIR, im) for im in os.listdir(MODEL_INTERPRETATION_IMAGES_DIR)]
    
    for im in im_paths:
        image = Image.open(im)
        st.image(image, caption=im.split('/')[-1])
