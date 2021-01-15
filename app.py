import streamlit as st
from ctgan import CTGANSynthesizer
from ctgan import load_demo
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sdv.metrics.tabular import CSTest, KSTest
from faker import Faker
from sdv.metrics.tabular import LogisticDetection, SVCDetection


le = LabelEncoder()
choices = ["Home","Conditional GANS"]

def df_cat(arg):
    df = arg
    cols = df.columns
    cat_columns = df._get_numeric_data().columns
    to_delete_columns = list(set(cols) - set(cat_columns))
    st.warning("Text columns detected: " + str(to_delete_columns))
    st.warning("Converting text columns to numerical format and removing any NaN rows: ")
    for i in to_delete_columns:
        df[i] = le.fit_transform(df[i].astype('str'))
    mydf = df
    mydf = mydf.dropna()
    return mydf

menu = st.sidebar.selectbox("Menu", choices)
st.set_option('deprecation.showfileUploaderEncoding', False)
if menu == "Conditional GANS":
    st.title("CGANS")
    st.write("Upload a CSV file (preferrably not too large) and use the command bar to start the magic. We suggest you drop the columns which you don\'t need before uploading.")
    st.write("--------------------------------------")
    file_upload = st.sidebar.file_uploader("Upload CSV")
    if file_upload:
        df = pd.read_csv(file_upload)
        st.write(df.head())
        newdf = df_cat(df)
        st.write(newdf.head())
        amount = st.sidebar.text_input("How many rows of data you need to generate?")
        #name = st.sidebar.text_input("Enter name of column with Name: ")
        columns = st.sidebar.text_input("Enter name of discrete column: ")
        discrete_columns = [columns]
        num_epochs = st.sidebar.slider("Number of epochs to train: ",20, 100)
        generate = st.sidebar.button("GENERATE")
        if generate:
            progress_bar = st.progress(0)
            for i in range(100):
                progress_bar.progress(i + 1)
            ctgan = CTGANSynthesizer(epochs=num_epochs)
            ctgan.fit(newdf, discrete_columns)
            samples = ctgan.sample(int(amount))
            st.success("Successfully generated {} rows of data".format(str(amount)))
            st.balloons()
            st.write(samples)
            ktestacc = KSTest.compute(newdf, samples)
            ldacc = LogisticDetection.compute(newdf, samples)
            st.write("KTest accuracy: " + str(ktestacc*100) + "%")
            st.write("**Logistic Detection** accuracy: " + str(ldacc*100) + "%")
elif menu == "Home":
    st.title("GAN based data generation")
    st.write("----------------------------------")
    st.header("Updates")
    st.write("**v1** of CGANS launched (2 days ago)")
    st.write("Visit the CGANS section from the menu and explore.")
    st.write('--------------------------------------')
    st.write("**v2** of CGANS launched (today)")
    st.write("Added conversion of text data to numerical data")