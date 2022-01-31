import os
import streamlit as st

class APPSettings():

    ########################     Global information    ########################
    
    title: str = "IA-PAU-DEMO"
    contacts: str = "moncoutiej@cy-tech.fr, urgellbapt@cy-tech.fr"
    root_dir: str = './' #os.path.join(os.path.dirname(__file__), os.pardir)
    model_interpretations_dir = os.path.join("app", 'assets', 'model-interpretation')  
    logo_surfriders: str = os.path.join("app", 'assets', 'logos', "logo_surf.png") #static/Logo_surfrider_fondation2020.png"
    logo_ia_pau: str = os.path.join("app", 'assets', 'logos', "logo_ia_pau.png")

@st.cache(suppress_st_warning=True)
def get_api_settings() -> APPSettings:
    """Init and return the APP settings

    Returns:
        APPSettings: The settings
    """
    return APPSettings()  # reads variables from environment