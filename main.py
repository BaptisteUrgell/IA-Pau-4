import streamlit as st
from app.page.multipage import MultiPage
from app.page import index


app = MultiPage()

# Title of the main page
st.title("IA PAU 2022")

# Add all your applications (pages) here
app.add_page("Home page",index.app)

# The main app
app.run()
