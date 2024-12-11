import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pickle

##load the model
model = tf.keras.models.load_model("model.h5")


##load the encoders and scaler

with open("onehot_encoder.pkl","rb") as file:
    onehot_encoder = pickle.load(file)


with open("label_encoder.pkl","rb") as file:
    label_encoder = pickle.load(file)

with open("scaler.pkl","rb") as file:
    scaler = pickle.load(file)


##streamlit app

st.title("Sales Price Prediction")


##user input

productcategory = st.selectbox("ProductCategory",onehot_encoder.categories_[0])
discount = st.number_input("Discount")
advertising_spend = st.number_input("AdvertisingSpend")
region = st.selectbox("Region",label_encoder.classes_)

#example for input data..

input_data = pd.DataFrame({
    "ProductCategory":[productcategory],
    "Discount":[discount],
    "AdvertisingSpend":[advertising_spend],
    "Region":[label_encoder.transform([region])[0]]

})

##one hot encodinf of Product category

onehot_encoding_category = onehot_encoder.transform([input_data["ProductCategory"]]).toarray()
onehot_encoding_df = pd.DataFrame(onehot_encoding_category,columns=onehot_encoder.get_feature_names_out(["ProductCategory"]))

df = pd.concat([input_data.drop("ProductCategory",axis=1),onehot_encoding_df],axis=1)


##scaling the input data
scaled_df = scaler.transform(df)


prediction = model.predict(scaled_df)
prediction_proba = prediction[0][0]

st.write(f'Sales Prediction Price: {prediction_proba}')
