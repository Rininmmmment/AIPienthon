import streamlit as st
import tensorflow
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import tensorflow as tf
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img, ImageDataGenerator
import pandas as pd
import math
train_dir = 'target_datasets/train' 
val_dir = 'target_datasets/val' 
backup_dir = './model' 
BATCH_SIZE = 32
IMAGE_SIZE = 64
val_data_gen = ImageDataGenerator(rescale=1./255)
model = load_model('my_model.h5')
# korekamoを名前に変換
def korekana(num):
  if num == 0:
    korename = "ツチガエル"
  elif num == 1:
    korename = "トノサマガエル"
  elif num == 2:
    korename = "アカメアマガエル"
  elif num == 3:
    korename = "モリアオガエル"
  elif num == 4:
    korename = "イチゴヤドクガエル"
  elif num == 5:
    korename = "アマガエル"
  return korename
st.set_page_config(
     page_title="ImgTest",
     page_icon="🧊",
     layout="centered",
     initial_sidebar_state="expanded"
 )






st.title("カエルを見分けよう")

reset = st.button("写真をリセットする")
if reset and os.path.isfile("target_datasets/val2/1/imported_file.jpg"):
    os.remove("target_datasets/val2/1/imported_file.jpg")
else:
    pass

st.write("")
uploaded_file=st.file_uploader("ファイルアップロード", accept_multiple_files=False)
if uploaded_file:
 image=Image.open(uploaded_file)
 img_array3 = np.array(image)
 img_array4 = np.expand_dims(image, axis=0)
 uploaded_predict = val_data_gen.flow(img_array4, y=None, batch_size=BATCH_SIZE)
 
 if not(os.path.isfile("target_datasets/val2/1/imported_file.jpg")):
    st.image(img_array3, width = 200)
 else:
     pass
 
 st.write("この写真で良いですか？")
 kousin = st.button('はい')
else:
 pass

if uploaded_file and kousin:
    # predict = tf.reshape(uploaded_predict, [-1])
    image.save("target_datasets/val2/1/imported_file.jpg")
else:
    pass

if os.path.isfile("target_datasets/val2/1/imported_file.jpg"):
    val_dir2 = "target_datasets/val2"
    predictdata = val_data_gen.flow_from_directory(
        val_dir2, target_size=(IMAGE_SIZE, IMAGE_SIZE),
        color_mode='rgb', batch_size=BATCH_SIZE,
        class_mode='categorical', shuffle=False)
    try:
        suiron = model.predict(predictdata, batch_size=BATCH_SIZE, verbose=0, steps=None)
        korekamo = np.argmax(suiron[0]) # 最も可能性が高いものはどれか格納
    except:
        pass
    
    st.write("### この写真は"+str(math.floor(suiron[0][korekamo]*100))+"%の確率で ---"+(str(korekana(korekamo))+"--- だよ。"))
    st.write("")
    st.write("↓　その他の推論結果")
    arr = [suiron[0][0], suiron[0][1], suiron[0][2], suiron[0][3], suiron[0][4], suiron[0][5]]
    label = ["Tsuchi"," Tono","Akame","Mori","Ichigo", "Ama"]
    fig, ax = plt.subplots()
    ax.bar(label, arr)
    st.pyplot(fig)
else:
    st.write("")
    st.write("画像をアップロードしてね！")
