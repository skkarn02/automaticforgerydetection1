# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 13:18:31 2022

@author: shilp
"""


import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.utils import normalize
from tensorflow.keras.models import *
import numpy as np
import pickle as pkl
from PIL import *


#Function for ela 
def convert_to_ela_image(image, quality):
    temp_filename = 'temp_file.jpg'
    ela_filename = 'temp_ela_file.png'
    
    image.save(temp_filename, 'JPEG', quality = quality)
    temp_image = Image.open(temp_filename)
    
    ela_image = ImageChops.difference(image, temp_image)
    
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    
    return ela_image

image_size = (128, 128)

def prepare_image(image_path):
    return np.array(convert_to_ela_image(image_path, 85).resize(image_size)).flatten() / 255.0


    
#Load model 
json_file = open('v1model.json', 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
# load weights into new model
model.load_weights("v1model.h5")

def predict(image,model) :
    im = Image.open(image)
    ela_img=prepare_image(im)
    ela_img=ela_img.reshape(1,128,128,3)
    prediction=model.predict(ela_img)
    
    return ela_img,prediction

      

st.title("Image Forgery Detection (Copy-Move Forgery Detection)")
st.header("Upload a image to get whether image is forged or pristine")
# To View Uploaded Image
image_file = st.file_uploader("Upload Images", type=["png","jpg"])
# You don't have handy image 
if bool(image_file)==True :
    st.image(image_file)
    ela_img,pred=predict(image_file,model)
    st.text("ELA image for this image")
    st.image(ela_img)
    pred=pred[0]
    if pred >= 0.5 :
        st.title("This is a pristine image")
    else :
        st.title("This is a fake image")
    st.write("##### NOTE : I am currently working to predict region where image is tampered . It would be Amazing !!!!. Please Stay Tuned . Thankyou ❤️")
        
else :
    ran_imageid=['Au_ani_00043','Au_sec_00040','Au_sec_30730','Tp_D_CRN_M_N_nat10129_cha00086_11522','Tp_D_CRN_S_N_cha10130_art00092_12187']
    st.text("")
    st.text("")
    st.text("You can download some sample images by clicking on the below links :")
    st.write("[link](https://drive.google.com/file/d/1yxQVw3e2znWid3i3KGJIT_MViun_X_nV/view?usp=sharing)")
    st.write("[link](https://drive.google.com/file/d/1bsgEQcLXjxOcdBMFNP7NRmOFhauPpL4K/view?usp=sharing)")
    st.write("[link](https://drive.google.com/file/d/1ccxMsGGKlJlXvGjmEFAcoYXx-N-bEASF/view?usp=sharing)")
    st.text("")
    st.text("")
    st.markdown("OOPS !!!!!!!!!! You are not ready with some images 😬. Don't worry i have some images for you click on the below button and it will predict whether random image is pristine or forged from a set of images. 😎")
    if st.button('Generate Caption for a random image') :
        ran_num=np.random.randint(0,len(ran_imageid))
        img_static_path=str(ran_imageid[ran_num])+'.jpg'
        img_temp=plt.imread(img_static_path)
        st.image(img_temp)
        ela_img,pred=predict(img_static_path,model)
        st.text("ELA image for this image")
        st.image(ela_img)
        pred=pred[0]
        if pred >= 0.5 :
            st.title("This is a pristine image")
        else :
            st.title("This is a fake image")
        st.text("")
        st.text("")
        st.text("")        
        st.write("##### NOTE : I am currently working to predict region where image is tampered . It would be Amazing !!!!. Please Stay Tuned . Thankyou ❤️")
            


    
