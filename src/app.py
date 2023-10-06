import streamlit as st
import pandas as pd
import pickle
from PIL import Image
import cv2
import base64

# Define a function to load the pickle file
def load_pickle(filepath):
    with open(filepath, "rb") as file:
        return pickle.load(file)

# Call the function to load your pickle file
data_dict = load_pickle("fund.-of-streamlit\src\streamlit_Basics.pkl")

# Now, you can access the components you saved in the dictionary.
scaler = data_dict["scaler"]
model = data_dict["model"]




img = cv2.imread(r'C:\Users\USER\Desktop\streamlit_app\fund.-of-streamlit\background_hp.jpg')
# Rotate the image 90 degrees clockwise
img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
st.image(img, channels='RGB')


    st.sidebar.title('Navigation')
    page = st.sidebar.radio('Select a Page', ['Home', 'Predict', 'Explore', 'About'])
    if page == 'Home':
        home()
    elif page == 'Predict':
        predict_sales()
    elif page == 'Explore':
        explore_data()
    elif page == 'About':
        about_info()

def home():

    def img_to_base64(img_path):
        with open(img_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')

    def set_css():
         img_data_uri = "data:image/jpg;base64," + img_to_base64('C:/Users/USER/Desktop/streamlit_app/fund.-of-streamlit/background_hp.jpg')
         st.markdown(f"""
        <style>
        body {{
        background-image: url("{img_data_uri}");
        background-size: cover;
        background-position: center;
        background-blend-mode: multiply;
        background-color: rgba(0,0,0,0.5);  /* adding the tint: in this case, a semi-transparent black */
        }} 
        </style>
    """, unsafe_allow_html=True)


    st.title("Welcome to Favorita Sales Prediction App")
    st.write("""
### To ensure the best of **Experience** on the Favorita Sales Prediction App,
Navigate using the sidebar(the arrow in the top left corner) to:
1. Predict sales
2. Explore the dataset
3. Learn more about this project.
""")

def predict_sales():
    st.title("Predict Sales")

    # Form for prediction
    with st.form(key="prediction_form"):
        holiday = st.selectbox("Is Today a Holiday :", ['Holiday', 'Not Holiday', 'Work Day', 'Additional', 'Event', 'Transfer','Bridge'])
        locale= st.radio("What holiday category does it fall under :", ['National', 'Not Holiday', 'Local', 'Regional'])
        Transferred = st.radio("Is the Holiday Transferred :", ['Yes', 'No'])
        onpromotion = st.number_input("What items on Promotion : ")
        
        # Submission of the form
        if st.form_submit_button("Predict"):
            try:
                df_input = pd.DataFrame({
                    "holiday": [holiday],
                    "locale": [locale],
                    "Transferred": [Transferred],
                    "onpromotion": [onpromotion]
                })

                # One-hot encode the data
                df_encoded = pd.get_dummies(df_input)

                # Ensuring the input data has the same columns as training data
                # Here, 'model_columns' is a list of columns used in the trained model.
                # This might come from the training data's columns after one-hot encoding.
                model_columns = ['onpromotion', 'holiday_Additional', 'holiday_Bridge', 'holiday_Event', 
                    'holiday_Holiday', 'holiday_Not Holiday', 'holiday_Transfer', 'holiday_Work Day', 
                    'locale_Local', 'locale_National', 'locale_Not Holiday', 'locale_Regional', 
                    'transferred_False', 'transferred_True']

                for col in model_columns:
                    if col not in df_encoded.columns:
                        df_encoded[col] = 0

                # Reorder columns based on model
                df_encoded = df_encoded[model_columns]

                # Now you can predict
                predictions = model.predict(df_encoded)

                # Display predicted sales
                st.balloons()
                st.write(f"The Predicted Sales is: ${predictions[0]}")

            except:
                st.write(f"Something went wrong during the Sales Prediction")

def explore_data():
    st.title("Explore Data")
    df = pd.read_csv(r"C:\Users\USER\Desktop\streamlit_app\fund.-of-streamlit\ts_data.csv")
    st.write(df)

def about_info():
    st.title("About")
    st.write("""## FAVORITA SALES PREDICTION APP
             
Beyond the bustling aisles and checkout lanes of grocery stores, there lies a complex web of purchasing patterns,
seasonal trends, and intricate sales dynamics. Understanding and predicting these dynamics can be the key to a retailer's success or downfall.
For Corporation Favorita, one of Ecuador's largest grocery retailers, these patterns translate to thousands of items across multiple store locations.
With the rise of data science and advanced analytical techniques,time series forecasting has emerged as an invaluable tool for businesses like Favorita.
We embarked on a journey to harness the power of time series forecasting,
aiming to craft a robust predictive model that can more accurately 
forecast the unit sales of Favorita's vast product range across its numerous stores. 
             
This app is a simple Streamlit application interface to a regression analysis done on The company Favorita sales data to predict sales.
Developed by Team Charleston during Azubi Cohort 4.
More details about the project and methodology can be found [here](https://github.com/florenceaffoh/LP3-TimeSeriesAnalysis.git).
""")

if __name__ == '__main__':
    main()
