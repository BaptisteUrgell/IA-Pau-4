import os
import streamlit as st

class APPSettings():

    ########################     Global information    ########################
    
    title: str = "IA-PAU-DEMO"
    contacts: str = "moncoutiej@cy-tech.fr, urgellbapt@cy-tech.fr"
    root_dir: str = './' #os.path.join(os.path.dirname(__file__), os.pardir)
    model_interpretations_dir = os.path.join(root_dir, 'assets', 'model-interpretation')  

@st.cache(suppress_st_warning=True)
def get_api_settings() -> APPSettings:
    """Init and return the APP settings

    Returns:
        APPSettings: The settings
    """
    return APPSettings()  # reads variables from environment