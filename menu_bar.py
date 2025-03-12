import streamlit as st
from streamlit_option_menu import option_menu

# Function to handle content display for each page
def display_page(page_name):
    pages = {
        "Home": ("Home Page", "Will be updated soon."),
        "About": ("About Page", "Will be updated soon."),
        "AI Coach": ("AI Coach Page", "Will be updated soon."),
        "Capstone Notes": ("Capstone Notes Page", "Will be updated soon."),
    }
    
    title, content = pages.get(page_name, ("Page Not Found", "This page does not exist."))
    st.title(title)
    st.write(content)

# Persistent Sidebar with Menu
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Home", "About", "AI Coach", "Capstone Notes"],
        icons=["house", "info-circle", "robot", "book"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal"
    )

# Main content is updated based on the selected menu item
display_page(selected)
