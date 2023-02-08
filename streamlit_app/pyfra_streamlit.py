import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

def run():
    st.set_page_config(
        page_title="Hello",
        page_icon="👋",
    )

    st.write("# pyfra 👋")
    st.image(image='images/stable_diffusion.jpeg', 
             width=200,
             caption='Generated with Stable Diffusion')

    with st.sidebar:
        add_radio = st.radio(
            "Project Pages",
            ("Page 1", "Page 2")
        )

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


if __name__ == "__main__":
    run()