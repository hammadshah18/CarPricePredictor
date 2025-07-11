

# 🚗 Car Price Predictor Web App

A simple and interactive **Streamlit** web application that predicts the market price of used cars in India based on various features like brand, model, fuel type, kilometers driven, and year of manufacturing.



---

## 🔍 Features

- Predicts used car prices using a trained machine learning model
- Dynamic dropdown for models based on the selected company
- Custom background image for a better UI
- Uses Streamlit for an easy-to-use web interface

---

## 📁 Project Structure

```

CarPricePredictor/
├── app.py                     # Main Streamlit app
├── car-bg.jpg                 # Background image
├── cars\_data.csv              # Cleaned dataset used for dropdowns
├── requirements.txt           # Python dependencies
├── readme.md                  # Project documentation
└── models/
└── LinearRegressionModel.pkl   # Trained ML model

````

---

## 🚀 How to Run Locally

1. **Clone the repo:**
   ```bash
   git clone https://github.com/hammadshah18/CarPricePredictor.git
   cd CarPricePredictor
````

2. **Create and activate a virtual environment (optional but recommended)**

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app:**

   ```bash
   streamlit run app.py
   ```

5. Open your browser at [http://localhost:8501](http://localhost:8501)

---

## 🌐 Live Demo

Coming soon on [Streamlit Cloud](https://streamlit.io/cloud)

---

## 🧠 Tech Stack

* **Python**
* **Pandas**
* **scikit-learn**
* **Joblib**
* **Streamlit**

---

## 📊 Model Info

* The model is a **Linear Regression** model trained on cleaned car listings data.
* Features used:

  * Company
  * Car Model
  * Manufacturing Year
  * Fuel Type
  * Kilometers Driven

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork the repo and submit a pull request.

---

## 📩 Contact

**Muhammad Hammad Shah**
📧 Email: \[[your-email@example.com](mailto:your-email@example.com)]
🔗 GitHub: [@hammadshah18](https://github.com/hammadshah18)





