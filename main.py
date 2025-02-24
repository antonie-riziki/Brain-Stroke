import streamlit as st 
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import missingno as msno
import autoreload
import warnings 
import time
import os
import sys
import csv

from collections import defaultdict, Counter

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

#%matplotlib inline

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 1000)

sb.set()
sb.set_style('ticks')


head_section = st.container()
data_loading = st.container()
graphical_rep = st.container()
feature_eng = st.container()
side_bar = st.container()


with head_section:
	st.image('200q.webp', width = 600)
	st.header('Brain Stroke Data Analysis')
	st.subheader('Overview')
	st.markdown("""

	Damage to the brain from interruption of its blood supply.
	
	A stroke is a medical emergency.
	
	### Symptoms of stroke include 
	
	- **trouble walking,** 
	- **speaking and understanding,**
	- **as well as paralysis or numbness of the face, arm or leg.**

	### Diagnosis
	Early treatment with medication like [tPA (clot buster)]() can minimise brain damage. Other treatments focus on limiting complications and preventing additional strokes.
	Over the years, we have been diagonising disease symptoms manual which led to misinformation of treatment, causing irreversible health problems
	
	### 
	Using [Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) ML Model to classify whether a person has brain stroke conditions or not.
	""")

with data_loading:
	df = pd.read_csv(r"D:\Open Classroom\Datasets\Brain Stroke Dataset\brain_stroke.csv")
	with st.spinner("Loading dataset..."):
		time.sleep(5)
		st.success("Done!")
	st.dataframe(df.head())

	st.markdown('statistical representation of data')
	
	col1, col2, col3 = st.columns(3)
	
	with col1:
		st.markdown('The dataset has ' + str(df.shape[0]) + ' features and ' + str(df.shape[1]) + ' records')
		st.write(df.describe())

	with col2:
		st.write(df.isnull().sum())

	with col3:
		st.markdown('We look at the Pearsons Correlation of the Dataset')
		fig = plt.figure()
		plt.title('Pearsons Correlation of Coefficients')
		sb.heatmap(df.corr(), annot = True, linewidth = 0.3)
		st.pyplot(fig)
	

	with graphical_rep:
		col1, col2 = st.columns(2)
		with col1:
			st.write('Distribution Plots')
			#def get_cols(data):
				#if i in data.dtypes != 'object':
					#return i
			cols = ['hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'stroke']
			col_select = st.selectbox('Select a series', df[cols].columns)
			dist_fig = plt.figure()
			sb.distplot(df[col_select])
			plt.title('Distribution Plot for ' + col_select)
			st.pyplot(dist_fig)

		with col2:
			st.write('Object Bar graphs')

			cols = ['ever_married', 'work_type', 'Residence_type', 'smoking_status', 'stroke']
			obj_cols = st.selectbox('Select a Series', df[cols].columns)
			bar_plt = plt.figure()
			sb.countplot(df[obj_cols], hue=df['gender'])
			plt.title('Popularity Distribution of ' + obj_cols + ' against Gender')
			st.pyplot(bar_plt)


with side_bar:
	with st.sidebar:
		st.markdown("""
			### Features
			""")
		st.image('img1.webp')

		gender = st.radio('Gender', ('Male', 'Female'))
		#gen_male = st.checkbox('female')

		age = st.slider('Select age', int(df['age'].min()), int(df['age'].max()))

		par_col1, par_col2, par_col3 = st.columns(3)

		with par_col1:
			hypertension = st.radio('Hypertension', ('Yes', 'No'))

		with par_col2:

			heart_disease = st.radio('Heart Disease', ('Yes', 'No'))

		with par_col3:

			ever_married = st.radio('Ever married', (df['ever_married'].unique()))

		avg_glc = st.slider('Average Glucose Level', float(df['avg_glucose_level'].min()), float(df['avg_glucose_level'].max()))

		work_type = st.selectbox('Work type', df['work_type'].unique())

		residence = st.selectbox('Residence type', df['Residence_type'].unique())

		smoking_status = st.selectbox('Smoking status', df['smoking_status'].unique())

		body_mass_index = st.slider('Body Max Index', int(df['bmi'].min()), int(df['bmi'].max()))

		# prediction_btn = st.button('Predict Stroke')

		# if "load state" not in st.session_state:
			# st.session_state.load_state



with feature_eng:
	le = LabelEncoder()

	def transform_categorical(data):
	    categories = (data.dtypes == 'object')
	    cat_cols = list(categories[categories].index)
	    le = LabelEncoder()
	    for col in cat_cols:
	        data[col] = le.fit_transform(data[col])

	transform_categorical(df)

	x = df.drop(columns = 'stroke')
	y = df['stroke']

	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=40)

	# print('x train', x_train.shape)
	# print('x test', x_test.shape)
	# print('y train', y_train.shape)
	# print('y test', y_test.shape)

	# Model Evaluation



	def run_experiment(model):

	    model.fit(x_train, y_train)
	    
	    pred = model.predict(x_test)
	    
	    conf_fig = plt.figure()
	    # plot_confusion_matrix(model, x_test, y_test, cmap = 'Blues')
	    plt.title('Confusion Matrix for ' + str(model))
	    plt.show()
	    #st.pyplot(conf_fig)
	    
	    # st.write(f'Precision {precision_score(y_test, pred)}')
	    # st.write(f'F1 Score {f1_score(y_test, pred)}')
	    # st.write(f'Recall {recall_score(y_test, pred)}')
	    # st.info(f'Model Accuracy \t\t {round(accuracy_score(y_test, pred) * 100, 2)}%')
	    with st.info(f'Model Accuracy \t\t {round(accuracy_score(y_test, pred) * 100, 2)}%'):
		    for i in range(int(round(accuracy_score(y_test, pred) * 100, 2))):
		        	time.sleep(0.0005)
		        	st.progress(i)
		    st.info(f'Model Accuracy \t\t {round(accuracy_score(y_test, pred) * 100, 2)}%')


	dtc = DecisionTreeClassifier()
	lr = LogisticRegression()
	gbc = GradientBoostingClassifier()
	rfc = RandomForestClassifier()

	models = {
	    dtc:DecisionTreeClassifier(),
	    lr:LogisticRegression(),
	    gbc:GradientBoostingClassifier(),
	    rfc:RandomForestClassifier()
	          }


	# select_model = st.selectbox('Model Selection', models)

	st.image('img2.jpg', width=650)

	# run_experiment(select_model)
	# if prediction_btn or session_state.load_state:
		# session_state.load_state = True

	def Manual_Testing(model, data):
	    input_data = data
	    
	    input_data_to_array = np.asarray(input_data)
	    
	    reshape_input_data = input_data_to_array.reshape(1, -1)
	    
	    model.fit(x_train, y_train)
	    
	    new_pred = model.predict(reshape_input_data)

	    #acc = accuracy_score(y_test, pred)

	    #st.write(f'Accuracy {accuracy_score(y_train, pred)}')
	    
	    if new_pred == 0:
	        st.success('Symptoms indicate no Brain damage. However stroke disease negative')
	        run_experiment(rfc)
	    

	    elif new_pred == 1:
	        st.warning('Alert!! Alert!! \nStroke Signs detected. Should seek professional medication immediatley.')
	        # run_experiment(rfc)


	# data = (1, 54.0, 0, 0, 1, 1, 1, 71.22, 28.5, 2)
	# Manual_Testing(rfc, data)

collect_data = {
	'gender': gender,	
	'age':	age,
	'hypertension':	hypertension,
	'heart_disease': heart_disease,	
	'ever_married':	ever_married,
	'work_type': work_type,
	'Residence_type': residence,	
	'avg_glucose_level': avg_glc,
	'bmi': body_mass_index,
	'smoking_status': smoking_status,
}

new_df = pd.DataFrame(collect_data, index = [0])
st.dataframe(new_df.head())

transform_categorical(new_df)
Manual_Testing(rfc, new_df)





		

		





