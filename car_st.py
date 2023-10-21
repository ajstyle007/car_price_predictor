import streamlit as st
import pickle
import numpy as np
import pandas as pd
from requests import request
model = pickle.load(open('car_price_predictor3.sav','rb'))

df = pd.read_csv("cleaned_car_data1.csv")

name_list = sorted(df["name"].unique())
cmpny_list = sorted(df["company"].unique())
year_list = sorted(df["year"].unique(), reverse=True)
kms_driven_list = sorted(df["kms_driven"].unique())
fuel_type_list = sorted(df["fuel_type"].unique())
 
def main(): 
     
    st.markdown("<h1 style='text-align: center; color: yellow;'>Car Price Predictor </h1>",  unsafe_allow_html=True)

 
   # st.title("Car Price Predictor")
    html_temp = """
    <div style = "background-color:teal;padding:10px">
    <h2 style = "color:black;text-align:center;"> Car Price Predictor ML App. </h2>
    </div>
    """
  
    st.markdown(html_temp, unsafe_allow_html=True )
    
    name = st.selectbox("What is the Car name?", name_list )  
    company = st.selectbox("What is the Car's company name?" , cmpny_list  )
    year = st.selectbox("Year of purchased?", year_list)
    #year = st.number_input("Car purchased year?" , min_value= 1995, max_value= 2019, step=1 )  
    kms_driven =  st.selectbox("Car km driven?" ,kms_driven_list  ) 
    fuel_type = st.selectbox("Fuel type of the Car?", fuel_type_list)

    user_data = {
        "name":name,
        "company":company,
        "year":year,
        "kms_driven":kms_driven,
        "fuel_type":fuel_type,
    }
    df_1 = pd.DataFrame.from_dict([user_data] )
     
    
    prediction = model.predict(df_1)

    button, price = st.columns(2)

    
    with button: 
        if st.button( "Predict.", type="primary" ): 
                
            with price:        
                result =  prediction 
                st.write("Approximate Car Predicted Price." )
                st.success('₹{}.'.format(np.round(int(result[0])),2))
                 
                #st.markdown(int(np.round(result),2))
         

if __name__ ==  "__main__":
    main()




