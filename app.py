import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)
import base64

def set_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    page_bg = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(page_bg, unsafe_allow_html=True)

# Call the function with your image filename
set_bg_from_local("car.jpg")


@st.cache_resource
def load_data():
    data_path = Path(__file__).parent / "new_cars.csv"
    df = pd.read_csv(data_path)
    companies = sorted(df['company'].unique())
    models = {c: sorted(df[df['company'] == c]['name'].unique()) for c in companies}
    return df, companies, models

@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "models" / "LinearRegressionModel.pkl"
    return joblib.load(model_path)

def main():
    df, COMPANIES, MODELS = load_data()
    model = load_model()

    st.title("🚗 Car Price Predicter")
    st.write("Predict the market value of used cars in Pakistan")

    with st.form("car_details"):
        st.subheader("Enter Car Details")

        col1, col2 = st.columns(2)
        with col1:
            company = st.selectbox("Manufacturer", COMPANIES, index=COMPANIES.index('Maruti') if 'Maruti' in COMPANIES else 0)
            # dynamically update model list when company changes
            model_list = MODELS.get(company, [])
            name = st.selectbox("Model", model_list, index=0 if model_list else -1)
            year = st.slider("Manufacturing Year", 2000, 2023, 2015)

        with col2:
            kms_driven = st.number_input("Kilometers Driven", min_value=0, max_value=500000, value=50000, step=1000)
            fuel_type = st.radio("Fuel Type", ["Petrol", "Diesel", "LPG"], index=0)

        submitted = st.form_submit_button("Predict Price")

        if submitted:
            if not name:
                st.error("Please select a model.")
                return

            input_data = pd.DataFrame([[name, company, year, kms_driven, fuel_type]],
                                      columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])
            try:
                prediction = model.predict(input_data)
                st.success(f"### Predicted Price: Pkr{prediction[0]:,.2f}")

                st.info(f"""
                - Manufacturer: {company}
                - Model: {name}
                - Year: {year}
                - Kilometers Driven: {kms_driven:,}
                - Fuel Type: {fuel_type}
                """)

            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

    st.markdown("---")
    
if __name__ == "__main__":
    main()
