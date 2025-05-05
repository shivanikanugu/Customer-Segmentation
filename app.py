import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import matplotlib.pyplot as plt
from sklearn import preprocessing

# Load the KMeans model
filename = 'kmeans.pkl'
try:
    loaded_model = joblib.load(filename)
    print(f"Model loaded successfully: {type(loaded_model)}")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

df = pd.read_csv("cluster_customer_data.csv")

st.markdown('<style>body{background-color:f0f0f0;}</style>', unsafe_allow_html=True)
st.title("Customer Segmentation using Machine Learning")

with st.sidebar:
    st.title("Input Data")
    balance = st.number_input("Balance", value=0.0, format="%.6f")
    balance_frequency = st.number_input("Balance Frequency", value=0.0, format="%.6f")
    purchases = st.number_input('Purchases', min_value=0.0, format="%.2f")
    oneoff_purchases = st.number_input('OneOff Purchases', min_value=0.0, format="%.2f")
    installments_purchases = st.number_input('Installments Purchases', min_value=0.0, format="%.2f")
    cash_advance = st.number_input('Cash Advance', min_value=0.0, format="%.6f")
    purchases_frequency = st.number_input('Purchases Frequency', min_value=0.0, step=0.01, format="%.6f")
    oneoff_purchases_frequency = st.number_input('OneOff Purchases Frequency', min_value=0.0, step=0.01, format="%.6f")
    purchases_installment_frequency = st.number_input('Purchases Installments Frequency', min_value=0.0, step=0.01, format="%.6f")
    cash_advance_frequency = st.number_input('Cash Advance Frequency', min_value=0.0, step=0.01, format="%.6f")
    cash_advance_trx = st.number_input('Cash Advance Trx', format="%d", min_value=0)
    purchases_trx = st.number_input('Purchases Trx', format="%d", min_value=0)
    credit_limit = st.number_input('Credit Limit', min_value=0.0, format="%.2f")
    payments = st.number_input('Payments', min_value=0.0, format="%.6f")
    minimum_payments = st.number_input('Minimum Payments', min_value=0.0, format="%.6f")
    prc_full_payment = st.number_input('PRC Full Payment', min_value=0.0, step=0.01, format="%.6f")
    tenure = st.number_input('Tenure', format="%d", min_value=1)
    
    input_data = [[balance, balance_frequency, purchases, oneoff_purchases, installments_purchases,
                   cash_advance, purchases_frequency, oneoff_purchases_frequency,
                   purchases_installment_frequency, cash_advance_frequency, cash_advance_trx,
                   purchases_trx, credit_limit, payments, minimum_payments, prc_full_payment, tenure]] 
    
    submitted = st.button("Submit")

if submitted:
    try:
        # Scale the input data
        scaler = preprocessing.StandardScaler()
        scaled_data = scaler.fit_transform(input_data)
        
        # Predict cluster
        cluster_label = loaded_model.predict(scaled_data)[0]
        st.success(f'Data belongs to Cluster: {cluster_label}')

        # Get data for the predicted cluster
        cluster_df = df[df['Cluster'] == cluster_label]

        st.subheader("Details of the selected cluster:")
        st.write(cluster_df)

        # Plot histograms for each attribute
        for attribute in cluster_df.columns:
            if attribute != 'Cluster':  # Skip the Cluster column
                hist_fig = px.histogram(cluster_df, x=attribute, color="Cluster", 
                                      title=f'Histogram of {attribute}')
                st.plotly_chart(hist_fig)

        # 3D scatter plot
        fig = px.scatter_3d(cluster_df, x='BALANCE', y='PURCHASES', z='PAYMENTS', 
                           color="Cluster",
                           title="3D Scatter Plot (Balance, Purchases, Payments)")
        st.plotly_chart(fig)
        
    except Exception as e:
        st.error(f"Error processing data: {e}")


   
