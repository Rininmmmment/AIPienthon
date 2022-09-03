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
train_dir = 'target_datasets/train' #trainフォルダをtrain_dirとして設定
val_dir = 'target_datasets/val' #valフォルダをval_dirとして設定
backup_dir = './model' #modelフォルダをbackup_dirとして設定
BATCH_SIZE = 32
IMAGE_SIZE = 64
val_data_gen = ImageDataGenerator(rescale=1./255)
model = load_model('puyo_model.h5')
# korekamoを名前に変換
def korekana(num):
  if num == 0:
    korename = "赤"
  elif num == 1:
    korename = "緑"
  elif num == 2:
    korename = "黄"
  elif num == 3:
    korename = "青"
  elif num == 4:
    korename = "紫"
  return korename
st.set_page_config(
     page_title="ImgTest",
     page_icon="🧊",
     layout="centered",
     initial_sidebar_state="expanded"
 )






st.title("ぷよぷよを見分けよう(デモ版)")

reset = st.button("写真をリセットする")
if reset and os.path.isfile("target_datasets/val2/1/imported_file.jpg"):
    os.remove("target_datasets/val2/1/imported_file.jpg")
else:
    pass

# ファイルアップロード
st.write("")
uploaded_file=st.file_uploader("ファイルアップロード (7KBまでなら正常にアップロードできます。)", accept_multiple_files=False)
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
    #推論結果を格納
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
    
    # 結果表示
    # st.image("target_datasets/val2/1/imported_file.jpg", width = 200)
    st.write("### この写真は"+str(math.floor(suiron[0][korekamo]*100))+"%の確率で ---"+(str(korekana(korekamo))+"ぷよ--- だよ。"))
    st.write("")
    st.write("↓その他の推論結果")
    arr = [suiron[0][0], suiron[0][1], suiron[0][2], suiron[0][3], suiron[0][4]]
    label = ["Red"," Green","Yellow","Blue","Purple"]
    fig, ax = plt.subplots()
    ax.bar(label, arr)
    st.pyplot(fig)
else:
    st.write("")
    st.write("画像をアップロードしてね！")
