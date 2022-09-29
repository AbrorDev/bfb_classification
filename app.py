import streamlit as st
from fastai.vision.all import *
import pathlib
import plotly.express as px
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# title
st.title("Ayiq, baliq va qushlarni klassifikatsiya qiluvchi dastur")

file = st.file_uploader('Rasm yuklash', type=['jpg', 'png', 'jpeg', 'gif', 'jfif'])
if file:
    st.image(file)

    img = PILImage.create(file)

    model = load_learner('bear_fish_bird.pkl')

    pred, pred_id, probs = model.predict(img)

    st.success(f"Bashorat: {pred}")
    st.info(f"Ehtimollik: {probs[pred_id]*100:.1f}%")

    fig = px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)