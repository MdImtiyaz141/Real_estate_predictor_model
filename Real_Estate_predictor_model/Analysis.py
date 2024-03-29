import streamlit as st
import pandas as pd
import plotly.express as px
st.set_page_config(page_title='Analysis')
#st.title('Analysis')
st.title('These Chart shows the map where there is more construction and the price')
df = pd.read_csv('C:/Users/IMTIYAZ/PycharmProjects/Real_estate_project/pages/chart_data.csv')
#st.dataframe(df)
fig = px.scatter_mapbox(df,color='avg_price',lat="lat", lon="long",size='SBA',
                  color_continuous_scale=px.colors.cyclical.IceFire,zoom=8.5,
                  mapbox_style='open-street-map',text='property_name')
st.plotly_chart(fig,use_container_width=True)

st.title('We are looking at the avg_price vs Super_built_up_Area')

fig1 = px.scatter(df,x='SBA',y='avg_price',hover_data='property_name')
st.plotly_chart(fig1,use_container_width=True)

