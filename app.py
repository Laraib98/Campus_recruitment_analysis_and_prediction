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
    return load('placement_prediction_model.pk')

st.set_page_config(
    page_title="Campus Recruitment Analysis and Prediction",
    page_icon="chart_with_upwards_trend"
)


st.subheader("Analysis or Predict")
option = st.selectbox('**Choose a page to view**',('Analysis','Predict'))
st.write('Analysis displays the data, analysis or vizualization.')
st.write('Prediction is for the prediction page.')


if option == 'Analysis':
    df = load_data()
    options1 = ['View data', 'View Analysis', 'View Visualization']
    options2 = ['Board of Secondary Education', 'Board of Higher Secondary Education', 'Undergraduate Degree', 'Postgraduate Specialisation', 'Work Experience', 
                        'Employability Test', 'Placement']

    st.header('MBA Student Placement Analysis')
    st.text("")
    ch = st.selectbox("**Select an option**", options1)

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
        
elif option == 'Predict':
    st.header('MBA Student Placement Prediction')
    st.markdown("**Select the details carefully**")
    st.text("")
    col1, col2 = st.columns(2, gap="large")

    with col1:
        gender = st.radio("**Select Gender** ", ('M', 'F'))
        st.text("")

        ssc_b = st.radio("**Select your SSC Board**", ('Central', 'Others'))
        st.text("")

        ssc_p = st.slider("**Select your SSC Percentage** ", 45.0, 100.0, format="%f%%", step=0.50)
        st.text('Selected: {}%'.format(ssc_p))
        st.text("")

        hsc_b = st.radio("**Select your HSC Board** ", ('Central', 'Others'))
        st.text("")

        hsc_s = st.radio("**Select your HSC Board Specialisation** ", ('Arts', 'Commerce', 'Science'))
        st.text("")

        hsc_p = st.slider("**Select your HSC Percentage** ", 45.0, 100.0, format ="%f%%",step=0.50)
        st.text('Selected: {}%'.format(hsc_p))
        st.text("")
        st.text("")
        st.text("")

        btnp= st.button("**Predict**")
       

    with col2:
        degree_t = st.radio("**Select your Undergraduate Degree Subject** ", ('Sci&Tech', 'Comm&Mgmt', 'Others'))
        st.text("")

        degree_p = st.slider("**Select your Undergraduate Degree Percentage** ", 45.0, 100.0, format="%f%%",step=0.50)
        st.text('Selected: {}%'.format(degree_p))
        st.text("")

        specialisation = st.radio("**Select your Postgraduate Degree Subject** ", ('Mkt&HR', 'Mkt&Fin'))
        st.text("")

        mba_p = st.slider("**Select your Postgraduate Degree Percentage** ", 45.0, 100.0, format="%f%%",step=0.50)
        st.text('Selected: {}%'.format(mba_p))
        st.text("")

        workex = st.radio("**Have Work Experience?** ", ('Yes', 'No'))
        st.text("")

        etest = st.slider("**Select your Employability Test Percentage** ", 45.0, 100.0, format="%f%%",step=0.50)
        st.text('Selected: {}%'.format(etest))

    if btnp:
        data = [gender, ssc_p, ssc_b, hsc_p, hsc_b,degree_p, workex, etest, specialisation, mba_p]
        # st.write(data)
        md = load_model()
        model = md.get('model')
        ordinal_enc = md.get('ordinal')
        hsc_enc = md.get('h_board')
        degree_enc = md.get('d_type')
        binary_vars = [gender[0],hsc_b, ssc_b, workex, specialisation]
        binary_vars = ordinal_enc.transform([binary_vars])   
        degree_dummy = degree_enc.transform([[degree_t]]).toarray()
        hsc_dummy = hsc_enc.transform([[hsc_s]]).toarray()
        x = pd.DataFrame([{
        'gender':  gender,
        'ssc_p':  ssc_p,
        'ssc_b':  ssc_b,
        'hsc_p':  hsc_p,
        'hsc_b':  hsc_b,
        'degree_p':  degree_p,
        'workex':  workex,
        'etest_p':  etest,
        'specialisation':  specialisation,
        'mba_p':  mba_p,   
        }])
        
        x[md.get('binary_col_names')] =   ordinal_enc.transform(x[md.get('binary_col_names')])
        xa = pd.concat([x, pd.DataFrame(hsc_dummy)], axis=1)
        xb = pd.concat([xa, pd.DataFrame(degree_dummy)], axis=1)
        # st.text(xb)
        out = model.predict(xb)
        if out[0] == 0:
            st.info("Not Placed")
        if out[0] == 1:
            st.info("Placed")









