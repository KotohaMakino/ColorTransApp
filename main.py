import streamlit as st
import cv2
import numpy as np
from PIL import Image
import requests
import io
from color_transformer import color_transform

# ---------- Streamlit UI ----------
st.write("画像の色変換サンプルです🌈") 
st.write("texturebase画像とcolorbase画像を選択すると、texturebase画像の色をcolorbase画像の色に変換できます") 

#  サンプル画像シャッフルボタン 
if st.button("サンプル画像をシャッフル"):
    st.session_state.sample_texture = Image.open(
        requests.get("https://picsum.photos/200/120", stream=True).raw
    )
    st.session_state.sample_color = Image.open(
        requests.get("https://picsum.photos/200/120", stream=True).raw
    )

# 画像読み込み
texturebase_file = st.file_uploader("Upload a texture-base image", type=["jpg", "jpeg", "png"])
if texturebase_file:
    image_texture = Image.open(texturebase_file)
else:
    image_texture = Image.open(requests.get("https://picsum.photos/200/120", stream=True).raw)

colorbase_file = st.file_uploader("Upload a color-base image", type=["jpg", "jpeg", "png"])
if colorbase_file:
    image_color = Image.open(colorbase_file)
else:
    image_color = Image.open(requests.get("https://picsum.photos/200/120", stream=True).raw)

# 色変換
color_transformed_image = color_transform(image_texture, image_color)

# 表示
tab1, tab2 = st.tabs(["Import Images", "Result"]) 
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.header("texture-base")
        st.image(image_texture, use_container_width=True)
    with col2:
        st.header("color-base")
        st.image(image_color, use_container_width=True)

with tab2:
    st.header("Result")
    st.image(color_transformed_image, use_container_width=True)
