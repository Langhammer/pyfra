import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from PIL import Image

def run():
    st.set_page_config(
        page_title="Hello",
        page_icon="👋",
    )

    pages = ["Project Presentation","Data Introduction and Cleaning","Data Visualization","Modeling","Further Analysis"]
    with st.sidebar:
        st.write('# Overview')
        page = st.radio(label='',options=pages)

    if page==pages[0]:
        st.header("PYFRA")
        st.write("# Car Accident Analysis in France")
        st.image(image='images/stable_diffusion.jpeg', 
                width=200,
                caption='Generated with Stable Diffusion')
        st.markdown(
            """
            PYFRA is a Data Science project, that analyzes goverment records of Car accidents in France
            between the year 2005 till 2021, with the aim of building a machine learning model to expect the 
            Severity of each car accident.
            
            ### Github
            [Project Page](https://github.com/DataScientest-Studio/pyfra)

            ### Project Members
            * Kay Langhammer
            * Robert Leisring
            * Saleh Saleh
            * Michal Turák

            ### Project Mentors
            * Robin Trinh
            * Laurene Bouskila
        """
        )

    if page==pages[1]:
        st.header('Data Introduction and Cleaning')
        st.markdown(
        
        '''
        ### Data Source
        The data used for this analysis is found in the official French goverment records and is split
         each year into 4 categories that include csv files of: Users, Characteristics, Vehicules, and Location.

        [Goverment Database](https://www.data.gouv.fr/en/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2021/)
        
        
        ### Data import and cleaning

        

        '''
        )

    if page==pages[2]:
        st.write('# Data Visualization')

        st.header("Hypothesis 1 : Accidents occur more often during a Specific time of the day and on weekends")
        image = Image.open("./figures/Accidents by Daytime.png")
        st.image(image)

        st.header("Hypothesis 2  : Accidents are split evenly among all age groups")
        image2 = Image.open("./figures/Accidents per Age.png")
        st.image(image2)
       
        st.header("Hypothesis 3  : Generally Accidents rarely result in serious injury or death")
        image3 = Image.open("./figures/Nb Accidents by Severity.png")
        st.image(image3)

        st.header("Hypothesis 4  : Severity of Accidents are split uniformly across all ages and sex groups")
        image4 = Image.open("./figures/Violin Chart.png")
        st.image(image4)

        st.header("Hypothesis 5  : Number of Accidents are higher for bad weather conditions")
        image5 = Image.open("./figures/Accidents by Weather conditions.png")
        st.image(image5)

        st.header("Hypothesis 6  : Number of Accidents generally should increase by year due to higher volume of traffic")
        image6 = Image.open("./figures/Accidents per Year.png")
        st.image(image6)



    if page==pages[3]:
        st.write('# Modeling')


        result_metrics = pd.read_pickle('./data/nb_3_results.p')

        st.write('## $f_1$ score by model')
        res_chart = alt.Chart(result_metrics).mark_bar().encode(
            x='f1',
            y="model"
        ).properties(height=300, width=500)
        
        # Plot the f1 scores
        st.altair_chart(res_chart)

        # Display the metrics table
        st.dataframe(data=result_metrics)


    if page==pages[4]:
        st.write('# Further Analysis')

if __name__ == "__main__":
    run()


