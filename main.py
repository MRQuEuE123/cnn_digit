import base64
from scipy.ndimage import center_of_mass
import math
import cv2
import numpy as np
#import os
import io
import streamlit as st
from PIL import Image
import pandas as pd

from tensorflow.keras.models import load_model
from pathlib import Path
from streamlit_drawable_canvas import st_canvas
model = load_model('D:\AI\MyOCR\Mycnn\digit_model.h5')


def getBestShift(img):
    cy, cx = center_of_mass(img)

    rows, cols = img.shape
    shiftx = np.round(cols / 2.0 - cx).astype(int)
    shifty = np.round(rows / 2.0 - cy).astype(int)

    return shiftx, shifty


def shift(img, sx, sy):
    rows, cols = img.shape
    M = np.float32([[1, 0, sx], [0, 1, sy]])
    shifted = cv2.warpAffine(img, M, (cols, rows))
    return shifted


def rec_digit(pil_image):

    img = np.array(pil_image)
    #img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = 255 - img
    # применяем пороговую обработку
    (thresh, gray) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # удаляем нулевые строки и столбцы
    while np.sum(gray[0]) == 0:
        gray = gray[1:]
    while np.sum(gray[:, 0]) == 0:
        gray = np.delete(gray, 0, 1)
    while np.sum(gray[-1]) == 0:
        gray = gray[:-1]
    while np.sum(gray[:, -1]) == 0:
        gray = np.delete(gray, -1, 1)
    rows, cols = gray.shape

    # изменяем размер, чтобы помещалось в box 20x20 пикселей
    if rows > cols:
        factor = 20.0 / rows
        rows = 20
        cols = int(round(cols * factor))
        gray = cv2.resize(gray, (cols, rows))
    else:
        factor = 20.0 / cols
        cols = 20
        rows = int(round(rows * factor))
        gray = cv2.resize(gray, (cols, rows))

    # расширяем до размера 28x28
    colsPadding = (int(math.ceil((28 - cols) / 2.0)), int(math.floor((28 - cols) / 2.0)))
    rowsPadding = (int(math.ceil((28 - rows) / 2.0)), int(math.floor((28 - rows) / 2.0)))
    gray = np.lib.pad(gray, (rowsPadding, colsPadding), 'constant')

    # сдвигаем центр масс
    shiftx, shifty = getBestShift(gray)
    shifted = shift(gray, shiftx, shifty)
    gray = shifted

    cv2.imwrite('D:\AI\MyOCR\Mycnn\gray.png', gray)
    #cv2.imwrite('gray' + img_path, gray)
    img = gray / 255.0
    img = np.array(img).reshape(-1, 28, 28, 1)
    out = str(np.argmax(model.predict(img)))
    return out


drawing_mode = st.sidebar.selectbox(
  #  "Drawing tool:", ("point", "freedraw", "line", "rect", "circle", "transform")
    "Drawing tool:", ( "freedraw", "transform")
)

stroke_width = st.sidebar.slider("Stroke width: ", 1, 45, 25)
stroke_color = st.sidebar.color_picker("Stroke color hex: ", "#000")
bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
#bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])

#realtime_update = st.sidebar.checkbox("Update in realtime", True)


# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    #background_image=Image.open(bg_image) if bg_image else None,
    #update_streamlit=realtime_update,
    height=500,
    width=900,
    drawing_mode=drawing_mode,
    #point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
    key="canvas",
)

my_text = " Результат распознавания:"

# Центрирование текста
result = st.button('Распознать',use_container_width=True)
#Do something interesting with the image data and paths
if canvas_result.image_data is not None:
    if result:
        saved_image = canvas_result.image_data
        pred =rec_digit(saved_image)
        #st.write(f"<div style='text-align: center;font-size: 48px;> {my_text}</div>", unsafe_allow_html=True)
        st.write(f"<div style='text-align: center;font-size: 72px;'>{pred}</div>", unsafe_allow_html=True)


