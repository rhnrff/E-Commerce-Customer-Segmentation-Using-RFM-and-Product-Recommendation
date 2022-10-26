import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity

scaler = pickle.load(open('scaler.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))
prod_vec_s = pd.read_csv('prod_vec_s.csv')

st.markdown("<h1 style='text-align: center; color: white;'>CITER</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: white;'>Customer-info Integration With Product Recommendation System</h1>", unsafe_allow_html=True)
st.image('https://cdn.discordapp.com/attachments/1032169809066078209/1033775113763573840/Black_Pink_Bold_Elegant_Monogram_Personal_Brand_Logo-1.png')

st.subheader('RFM Customer Segmentation')
recency = st.number_input('How Many Days Ago Was The Last Purchase Done By The Customer')
frequency = st.number_input('How Many Times Has The Customer Done Transaction')
monetary = st.number_input('How Much Has The Customer Spent On The Store')

if st.button('Submit'):

    x = pd.DataFrame([[recency, frequency, monetary]],
    columns = ['recency', 'frequency', 'monetary'])
    xs = scaler.transform(x)
    pred = model.predict(xs)

    if pred == 2:
        prediction = 'Loyal Customers'
    elif pred == 1:
        prediction = 'Middle Customers'
    elif pred == 3:
        prediction = 'Low Customers'
    else:
        prediction = 'Near Lost'
    
    st.write(prediction)

st.subheader('Product Recommendation')
uploaded_img = st.file_uploader('Upload The Last Purchased Product Image')
if uploaded_img != None:
    image_name = uploaded_img.name

    def show_img(image_name,title=image_name):
        img_path = 'images_s/'+str(image_name)
        im = cv2.imread(img_path)
        fig = plt.figure()
        plt.axis('off')
        plt.imshow(im[:,:,::-1])
        plt.title(title)
        st.pyplot(fig)

    def fetch_most_similar_products(image_name,Cust_n,n_similar=3):
        st.text("-----------------------------------------------------------------------")
        st.text("Original Product:")
        show_img(image_name,image_name)
        prod_vec_se=prod_vec_s
        prod_vec_se.loc[prod_vec_se['image_x']==image_name, 'class']= Cust_n
        cosine_similarity_df = pd.DataFrame(cosine_similarity(prod_vec_se.drop('image_x',axis=1)))
        curr_index = prod_vec_se[prod_vec_se['image_x']==image_name].index[0]
        closest_image = pd.DataFrame(cosine_similarity_df.iloc[curr_index].nlargest(n_similar+1)[1:])
        st.text("-----------------------------------------------------------------------")
        st.text("Recommended Product")
        for index,imgs in closest_image.iterrows():
            similar_image_name = prod_vec_se.iloc[index]['image_x']
            similarity = np.round(imgs.iloc[0],3)
            show_img(similar_image_name,str(similar_image_name)+' nSimilarity : '+str(similarity))

if st.button('Recommend'):

    x = pd.DataFrame([[recency, frequency, monetary]],
    columns = ['recency', 'frequency', 'monetary'])
    xs = scaler.transform(x)
    pred = model.predict(xs)
    if uploaded_img != None:
        fetch_most_similar_products(image_name,(pred*10))
