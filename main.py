import streamlit as st
from app.page.multipage import MultiPage
from app.page import index, demonstration, model_interpretation, pollution_analysis


app = MultiPage()

# Title of the main page
st.title("IA PAU 4 - CY Riders")

# Add all your applications (pages) here
app.add_page("Demonstration",demonstration.app)
app.add_page("Model Interpretation",model_interpretation.app)
app.add_page("Energy Cost Analysis",pollution_analysis.app)

# The main app
app.run()
