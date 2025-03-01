import streamlit as st
from streamlit_option_menu import option_menu

# Navigation menu
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Home", "About", "AI Coach", "Capstone Notes"],
        icons=["house", "info-circle", "robot", "book"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal"
    )

# Load the selected page
if selected == "Home":
    st.title("Home Page")
    st.write("Will be updated soon.")
elif selected == "About":
    st.title("About Page")
    st.write("Will be updated soon.")
elif selected == "AI Coach":
    st.title("AI Coach Page")
    st.write("Will be updated soon.")
elif selected == "Capstone Notes":
    st.title("Capstone Notes Page")
    st.write("Will be updated soon.")