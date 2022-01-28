import streamlit as st
from app.page.multipage import MultiPage
from app.page import index, demonstration


app = MultiPage()

# Title of the main page
st.title("IA PAU 4")

# Add all your applications (pages) here
app.add_page("Home page",index.app)
app.add_page("Demonstration",demonstration.app)

# The main app
app.run()
