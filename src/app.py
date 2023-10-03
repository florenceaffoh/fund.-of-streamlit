import streamlit as st

st.write("""### My Interface
This is my first interface using streamlit
""")
         
import streamlit as st
import pandas as pd
import pickle

# Load ML models and components
st.cache #To prevent the reruning of the app anytime there's a change
def load_ml_components(fp):
    with open(fp, "rb") as file:
        return pickle.load(file)
   

ml_component_dict = load_ml_components(fp="src\assets\ml\ml_components.pkl")
end2end_pipeline = ml_component_dict['pipeline']
labels = ml_component_dict['labels']

def main():
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
    st.title("Welcome to Favorita Sales Prediction App")
    st.write("""
### To ensure the best of **Experience** on the Favorita Sales Prediction App,
Navigate using the sidebar to:
1. Predict sales
2. Explore the dataset
3. Learn more about this project.
""")

def predict_sales():
    st.title("Predict Sales")
   
   # Getting user input
    holiday = st.radio("Is Today a Holiday :", ['Yes', 'No'])

    # Add more widgets for other columns...

    if st.button("Predict"):
        # Create dataframe from input
        df_input = pd.DataFrame({
            "holiday": [holiday],
            # ... add other columns
        })

        # Predict using ML pipeline
        predictions = end2end_pipeline.predict(df_input)

        # Display results
        st.write(f"Predicted Sales: {predictions[0]}")

def explore_data():
    st.title("Explore Data")
    df = pd.read_csv("path_to_data.csv")
    st.write(df)

def about_info():
    st.title("About")
    st.write("""
    This is a simple Streamlit application to predict sales.
    Developed by [Your Name].
    More details about the project and methodology can be found [here](your_link).
    """)

if __name__ == '__main__':
    main()


