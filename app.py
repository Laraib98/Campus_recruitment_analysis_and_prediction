from optparse import Values
from statistics import mode
import streamlit as st
from joblib import load
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

# scikit-learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_validate

def status_to_number(i):
    if i == 'Placed':return 1
    elif i == 'Not Placed':return 0
    else: return i

st.cache()
def load_data():
    df=pd.read_csv('Placement_Data_Full_Class.csv')
    df['status_num'] = df['status'].apply(status_to_number)
    return df
    

def load_model():
    return load('model/campus_recruitment_model_v1.jb')

st.set_page_config(
    page_title="Placement Prediction",
    page_icon="chart_with_upwards_trend"
)

with st.sidebar:
    st.subheader("Analysis or Predict")
    option = st.selectbox('Choose a page to view',('Analysis','Predict'))
    st.write('Analysis displays the data, analysis or vizualization.')
    st.write('Prediction is for the prediction page.')



df = load_data()
options1 = ['View data', 'View Analysis', 'View Visualization']
options2 = ['Board of Secondary Education', 'Board of Higher Secondary Education', 'Undergraduate Degree', 'Postgraduate Specialisation', 'Work Experience', 
                      'Employability Test', 'Placement']

st.header('MBA Student Placement Analysis')

ch = st.selectbox("select an option", options1)

if ch == options1[0]:
    st.dataframe(df)
if ch == options1[1]:
    st.write(df.describe())
    st.info("Unique value count in each columns")
    st.write(df.nunique())
if ch == options1[2]:
    choice = st.selectbox("View Visualization: ", options2)
    if choice == options2[0]:
        fig = px.pie(df, names='ssc_b', title='Distribution of Board of Secondary Education')
        fig.update_layout(title_x=0.5)
        st.plotly_chart(fig, use_container_width=True)
        ssc_b_gender_df = pd.pivot_table(df, index='ssc_b', columns='gender',aggfunc='count')
        gender_ssc_b_df = ssc_b_gender_df.degree_p.reset_index()
        fig1 = px.bar(gender_ssc_b_df, gender_ssc_b_df.ssc_b, ['M','F'])
        fig1.update_layout(title='Gender Distribution(By Board of Secondary Education)', title_x=0.5)
        st.plotly_chart(fig1, use_container_width=True)
        fig2 = px.histogram(df,x='ssc_p', histfunc='count', marginal='box', title='Distribution of Secondary Education Percentage')
        fig2.update_layout(title_x=0.5)
        st.plotly_chart(fig2, use_container_width=True)
    if choice == options2[1]:
        pass

    if choice == options2[1]:
        fig = px.pie(df,names='hsc_b',title ='Distribution of Board of Higher Secondary Education')
        fig.update_layout(title_x=0.5)
        st.plotly_chart(fig, use_container_width=True)
        fig1 = px.pie(df,names='hsc_s', title='Distribution of Higher Secondary Education Subjects')
        fig1.update_layout(title_x=0.5)
        st.plotly_chart(fig1, use_container_width=True)
        fig2 = px.histogram(df,x='hsc_p', marginal='box', title='Distribution of Higher Secondary Education Percentage')
        fig2.update_layout(title_x=0.5)
        st.plotly_chart(fig2, use_container_width=True)
    if choice == options2[2]:
        pass

    if choice == options2[2]:
        fig = px.pie(df,names='degree_t',title='Distribution of Undergraduate Degree Subject')
        fig.update_layout(title_x=0.5)
        st.plotly_chart(fig, use_container_width=True)
        fig1 = px.histogram(df,x='degree_p', marginal='box', title='Distribution of Undergraduate Degree Percentage')
        fig1.update_layout(title_x=0.5)
        st.plotly_chart(fig1, use_container_width=True)
    if choice == options2[3]:
        pass

    if choice == options2[3]:
        fig = px.pie(df,names='specialisation', title='Distribution of Postgraduate Subject')
        fig.update_layout(title_x=0.5)
        st.plotly_chart(fig, use_container_width=True)
        fig1 = px.histogram(df,x='mba_p', marginal='box', title='Distribution of Postgraduate Percentage')
        fig1.update_layout(title_x=0.5)
        st.plotly_chart(fig1, use_container_width=True)
        salary_spc = df.groupby('specialisation')['salary'].sum().reset_index()
        fig2 = px.pie(salary_spc, 'specialisation','salary')
        fig2.update_layout(title='Overall Salary Distribution(By Specialisation)', title_x=0.5)
        st.plotly_chart(fig2, use_container_width=True)

    if choice == options2[4]:
        pass

    if choice == options2[4]:
        fig = px.pie(df,names='workex', title='Distribution of Work Experience')
        fig.update_layout(title_x=0.5)
        st.plotly_chart(fig, use_container_width=True)
        gender_exp_df = pd.pivot_table(df, index='workex', columns='gender',aggfunc='count')
        workex_gender_df = gender_exp_df.degree_p.reset_index()
        fig1 = px.bar(workex_gender_df, workex_gender_df.workex, ['M','F'])
        fig1.update_layout(title='Gender Distribution(By Work Experience)', title_x=0.5)
        st.plotly_chart(fig1, use_container_width=True)
    if choice == options2[5]:
        pass

    if choice == options2[5]:
        fig = px.histogram(df,x='etest_p', marginal='box', title='Employability Test Percentage')
        fig.update_layout(title_x=0.5)
        st.plotly_chart(fig, use_container_width=True)
    if choice == options2[6]:
        pass

    if choice == options2[6]:
        fig = px.pie(df,names='status', title='Distribution of Placement')
        fig.update_layout(title_x=0.47)
        st.plotly_chart(fig, use_container_width=True)
        placement_df = pd.pivot_table(df, index='status', columns='gender',aggfunc='count')
        degree_placement_df = placement_df.degree_p.reset_index()
        fig1 = px.bar(degree_placement_df, degree_placement_df.status, ['F','M'])
        fig1.update_layout(title='Placement Distribution(By Gender)', title_x=0.5)
        st.plotly_chart(fig1, use_container_width=True)
        placement_df = pd.pivot_table(df, index='status', columns='specialisation',aggfunc='count')
        placement_specialisation_df = placement_df.degree_p.reset_index()
        fig2 = px.bar(placement_specialisation_df, placement_specialisation_df.status, ['Mkt&HR','Mkt&Fin'])
        fig2.update_layout(title='Placement Distribution(By Specialisation)', title_x=0.5)
        st.plotly_chart(fig2, use_container_width=True)
        workex_status = df.groupby('status')['workex'].count().reset_index()
        fig3 = px.pie(workex_status, 'status','workex')
        fig3.update_layout(title='Placement Distribution(By Work Experience)', title_x=0.5)
        st.plotly_chart(fig3, use_container_width=True)
    
