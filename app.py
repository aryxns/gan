import streamlit as st
from ctgan import CTGANSynthesizer
from ctgan import load_demo
import pandas as pd
from sdv.metrics.tabular import CSTest, KSTest
from sdv.metrics.tabular import LogisticDetection, SVCDetection

choices = ["Home","Conditional GANS"]
menu = st.sidebar.selectbox("Menu", choices)

if menu == "Conditional GANS":
    st.title("CGANS")
    st.write("Upload a CSV file (preferrably not too large) and use the command bar to start the magic.")
    st.write("--------------------------------------")
    file_upload = st.sidebar.file_uploader("Upload CSV")
    if file_upload:
        df = pd.read_csv(file_upload)
        st.write(df.head())
        amount = st.sidebar.text_input("How many rows of data you need to generate?")
        columns = st.sidebar.text_input("Enter name of discrete column: ")
        discrete_columns = [columns]
        num_epochs = st.sidebar.slider("Number of epochs to train: ",20, 100)
        generate = st.sidebar.button("GENERATE")
        if generate:
            st.progress(100)
            ctgan = CTGANSynthesizer(epochs=num_epochs)
            ctgan.fit(df, discrete_columns)
            samples = ctgan.sample(100)
            st.success("Successfully generated {} rows of data".format(str(amount)))
            st.balloons()
            st.write(samples.head())
            ktestacc = KSTest.compute(df, samples)
            ldacc = LogisticDetection.compute(df, samples)
            st.write("KTest accuracy: " + str(ktestacc*100) + "%")
            st.write("Logistic Detection accuracy: " + str(ldacc*100) + "%")
elif menu == "Home":
    st.title("GAN based data generation")
    st.write("----------------------------------")
    st.header("Updates")
    st.write("**v1** of CGANS launched (today)")
    st.write("Visit the CGANS section from the menu and explore.")