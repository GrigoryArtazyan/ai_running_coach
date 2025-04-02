import matplotlib
matplotlib.use('Agg')  # Ensures Matplotlib uses a backend compatible with Streamlit

import streamlit as st
import tempfile
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import os
import sys
from streamlit_vertical_slider import vertical_slider
from streamlit_option_menu import option_menu

import menu_bar
