import streamlit as st
import pandas as pd
import numpy as np
#Added later
import plotly.express as px
print('Cheked!!!')



st.set_page_config(page_title='Basic Sales Dashboard', layout='wide')

# generate sample data
np.random.seed(42)
df = pd.DataFrame({
    'Date':pd.date_range('2024-01-01', periods=100), 
    'Sales': np.random.randint(500, 2000, size=100),
    'Region': np.random.choice(['North', 'South', 'East', 'West'], size=100),
    'Product': np.random.choice(['Product A', 'Product B', 'Product C'], size=100)
})

#SideBar Filters
st.sidebar.title('Filter')
regions = st.sidebar.multiselect('Select Region', df['Region'].unique(), default=df['Region'].unique())
products = st.sidebar.multiselect('Select Product', df['Product'].unique(), default=df['Product'].unique())

#Filter Data
filtered_df = df[(df['Region'].isin(regions)) & (df['Product'].isin(products))]

#Display metrics
col1, col2, col3 = st.columns(3)
col1.metric('Total Sales', f"${filtered_df['Sales'].sum():,}")
col2.metric('Average Sales', f"${filtered_df['Sales'].mean():.0f}")
col3.metric('Records', len(filtered_df))



#Charts
col1, col2, col3 = st.columns(3)

with col1:
    fig_line = px.line(filtered_df, x='Date', y='Sales', color='Region', title='Sales Over Time')
    st.plotly_chart(fig_line, use_container_width=True)

with col2:
    region_sales = filtered_df.groupby('Region')['Sales'].sum().reset_index()
    fig_bar = px.bar(region_sales, x='Region', y='Sales', title='Total Sales By Region')
    st.plotly_chart(fig_bar, use_container_width=True)

with col3: # Pie chart for product distribution
    fig_pie = px.pie(filtered_df, values='Sales', names='Product', title='Sales by Product')
    st.plotly_chart(fig_pie)

# Data Table
#Display Filtered Data
st.subheader('Filtered Data')
st.dataframe(filtered_df)


