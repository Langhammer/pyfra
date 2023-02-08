import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

def run():
    st.set_page_config(
        page_title="Hello",
        page_icon="👋",
    )

    pages = ['Page 0', 'Page 1', 'Page 2']
    with st.sidebar:
        st.write('# Overview')
        page = st.radio(label='',options=pages)

    if page==pages[0]:
        st.write("# pyfra")
        st.image(image='images/stable_diffusion.jpeg', 
                width=200,
                caption='Generated with Stable Diffusion')
        st.markdown(
            """
            pyfra is a Data Science project.
            Text...
            ### Github
            [Project Page](https://github.com/DataScientest-Studio/pyfra)

            ### Project Members
            * Kay Langhammer
            * Robert Leisring
            * Saleh Saleh
            * Michal Turák
        """
        )

    if page==pages[1]:
        st.write('# Page 1')

    if page==pages[2]:
        st.write('# Page 2')

if __name__ == "__main__":
    run()