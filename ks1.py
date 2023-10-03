import streamlit as st
import pandas as pd
import pickle
import os
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, explained_variance_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from math import sqrt
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import warnings

def load_ml_component(fp):
    "load the ml component to reuse in app"
    with open(fp,"rb"):
        object = pickle.load(fp)
    return object


# Setup
## Variable and Constant
DIRPATH  = os.path.dirname(os.path.realpath('C:\Users\USER\Desktop\fund. of streamlit\To_Export_file.ipynb'))
ml_core_fp = os.path.join(DIRPATH, "assets", "ml", "ml_components.pkl")

# Execution
ml_component_dict = load_ml_component(fp = ml_core_fp)

labels = ml_component_dict['labels']
idx_to_labels = {i: l for(i,l) in enumerate('labels')} 

end2end_pipeline = ml_component_dict['pipeline']


print(f"\n[Info] Predictable Lables : {labels}")
print(f"\n[Info] Indexes to Lables : {idx_to_labels}")
print(f"\n[Info] ML components loaded: {list(ml_component_dict.keys())}")


#Interface


# Input
holiday = st.text_input("Is Today a Holiday :")
locale = st.text_input("What type of holiday is it: ")
transferred = st.text_input("Is this a transferred holiday")
store_number = st.text_input("What's the store number")


# Prediction Execution
if st.button("Predict"):
    try:
        #Dataframe Creation
        df = pd.DataFrame(
            {
            "holiday":[holiday], "locale":[locale], "transferred":[transferred]
            }
        )

        #ML PART
        output = end2end_pipeline.predict_proba(df)

        #store confidence score/prediction for prediction class
        confidence_score = output.max(axis=1)
        df['confidence score'] = output.max(axis=1)

        # get index of predicted class
        predicted_idx = output.argmax(axis = 1)

        # store index then replace by the matching label

        df['predicted label'] =  predicted_idx
        predicted_label = df['predicted label'] .replace(idx_to_labels)
        df['predicted label'] = predicted_label




        print(f"(Info) Input data as dataframe: \n{df.to_markdown()}")
        #df.columns = num_cols + cat_cols
        #df = df(num_cols + cat_cols) #reorder the dataframe
        
        st.balloons()
        st.success(f"The sales for store {store_number} is: '{predicted_label[0]} with a confidence score of {confidence_score[0]}'")

    except:
        st.error("Something went wrong during the sales prediction.")   

   