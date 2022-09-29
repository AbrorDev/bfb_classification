import streamlit as st
from fastai.vision.all import *
import pathlib
import plotly.express as px
import platform

plt = platform.system()
if plt=='Linux':
    pathlib.WindowsPath = pathlib.PosixPath

# title
st.title("Ayiq, baliq va qushlarni klassifikatsiya qiluvchi dastur")

file = st.file_uploader('Rasm yuklash', type=['jpg', 'png', 'jpeg', 'gif', 'jfif'])
if file:
    st.subheader("Yuborilgan rasm")
    st.image(file)

    img = PILImage.create(file)

    model = load_learner('bear_fish_bird.pkl')

    pred, pred_id, probs = model.predict(img)

    st.success(f"Bashorat: {pred}")
    st.info(f"Ehtimollik: {probs[pred_id]*100:.1f}%")

    fig = px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)
    st.write("Qaysi sinfga necha foiz o'xshashligining grafik ko'rinishi")
