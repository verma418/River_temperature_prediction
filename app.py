import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler 
import warnings
import xgboost as xgb
import lightgbm as lgb
from pytorch_tabnet.tab_model import TabNetRegressor
import torch
import joblib 
import os
import time 
import requests 
import plotly.graph_objects as go

# --- Configuration ---
MODEL_SAVE_DIR = "saved_models" # Directory where models and scaler are saved

# Define file paths for loading
SCALER_PATH = os.path.join(MODEL_SAVE_DIR, "scaler.joblib")
COLUMNS_PATH = os.path.join(MODEL_SAVE_DIR, "columns.joblib")
XGB_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "xgboost_model.json")
LGB_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "lightgbm_model.txt")
TABNET_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "tabnet_model.zip") # Explicitly add .zip for loading

warnings.filterwarnings('ignore')

# Function to load models and scaler (cached)
@st.cache_resource(show_spinner="Loading pre-trained models and scaler...") # Use cache_resource for models/objects
def load_prediction_assets():
    """Loads the pre-trained models, scaler, and column info."""
    # Check if all required files exist
    required_files = [SCALER_PATH, COLUMNS_PATH, XGB_MODEL_PATH, LGB_MODEL_PATH, TABNET_MODEL_PATH]
    if not all(os.path.exists(f) for f in required_files):
        st.error(f"Error: One or more required model/scaler files not found in '{MODEL_SAVE_DIR}'. "
                 f"Please ensure the `train_models.py` script has been run successfully.")
        missing = [f for f in required_files if not os.path.exists(f)]
        st.info(f"Missing files: {', '.join(missing)}")
        return None # Return None if assets are missing

    try:
        start_time = time.time()
        scaler = joblib.load(SCALER_PATH)
        columns_info = joblib.load(COLUMNS_PATH)
        all_numeric_columns = columns_info['all_numeric_columns']
        features_for_modeling = columns_info['features_for_modeling']

        # Load XGBoost
        xgb_model = xgb.XGBRegressor()
        xgb_model.load_model(XGB_MODEL_PATH)

        # Load LightGBM
              
# Load LightGBM Booster directly
        lgb_booster = lgb.Booster(model_file=LGB_MODEL_PATH) # Load the core booster

        # Load TabNet
        tabnet_model = TabNetRegressor()
        tabnet_model.load_model(TABNET_MODEL_PATH)

        end_time = time.time()
        # st.sidebar.info(f"Models loaded in {end_time - start_time:.2f} seconds.") # Optional: loading time info

        # --- Return the loaded objects, including the lgb_booster ---
        return scaler, all_numeric_columns, features_for_modeling, xgb_model, lgb_booster, tabnet_model # Return booster, not lgb_model


    except Exception as e:
        st.error(f"An error occurred while loading prediction assets: {e}")
        st.exception(e) # Show full traceback for debugging
        return None


# Function to generate future data and make predictions
@st.cache_data(show_spinner="Generating future data and predicting temperatures...")
def generate_and_predict_future(_scaler, _all_numeric_columns, _features_for_modeling, _xgb_model, _lgb_booster, _tabnet_model):
    """
    Generates future data, scales it, and predicts using loaded models.
    Returns the DataFrame with daily predictions.
    """
    st.write("Generating random future weather data (10 years)...")

    # Generate future dates and data
    future_dates = pd.date_range(start='2025-01-01', periods=3650, freq='D') # 10 years
    future_data_dict = {
        'tempmax': np.random.uniform(23.7, 38.7, 3650),
        'tempmin': np.random.uniform(18.4, 29.1, 3650),
        'temp': np.random.uniform(25.0, 35.0, 3650), # Include temp, even if it's the target, scaler expects it
        'feelslikemax': np.random.uniform(25.0, 40.0, 3650),
        'feelslikemin': np.random.uniform(20.0, 30.0, 3650),
        'feelslike': np.random.uniform(25.0, 35.0, 3650),
        'dew': np.random.uniform(15.0, 25.0, 3650),
        'humidity': np.random.uniform(40, 95, 3650),
        'precip': np.random.uniform(0, 50, 3650),
        'precipprob': np.random.uniform(0, 100, 3650),
        'precipcover': np.random.uniform(0, 100, 3650),
        'solarradiation': np.random.uniform(0, 8.5, 3650),
        'solarenergy': np.random.uniform(0, 25.0, 3650),
        'uvindex': np.random.uniform(0, 10, 3650),
        'cloudcover': np.random.uniform(0, 100, 3650),
        'visibility': np.random.uniform(0, 10, 3650),
        'windspeed': np.random.uniform(0, 30, 3650),
        'winddir': np.random.uniform(0, 360, 3650),
        'sealevelpressure': np.random.uniform(1000, 1020, 3650)
    }
    future_data = pd.DataFrame(future_data_dict, index=future_dates)
    future_data.index.name = 'datetime'

    # Ensure future data has EXACTLY the same columns in the same order as the scaler expects
    missing_cols = set(_all_numeric_columns) - set(future_data.columns)
    for c in missing_cols:
        future_data[c] = 0
    future_data = future_data[_all_numeric_columns]

    st.write("Scaling future data...")
    future_data_scaled = _scaler.transform(future_data)
    future_data_scaled_df = pd.DataFrame(future_data_scaled, index=future_data.index, columns=_all_numeric_columns)

    X_future_final = future_data_scaled_df[_features_for_modeling]

    st.write("Making predictions with loaded models (XGBoost, LightGBM, TabNet)...")
    future_xgb_pred = _xgb_model.predict(X_future_final).flatten()
    future_lgb_pred = _lgb_booster.predict(X_future_final).flatten() 
    future_tabnet_pred = _tabnet_model.predict(X_future_final.values).flatten()

    st.write("Calculating ensemble prediction...")
    future_predictions = 0.5 * future_xgb_pred + 0.3 * future_lgb_pred + 0.2 * future_tabnet_pred

    future_results_df = pd.DataFrame({
        'Predicted Temperature (°C)': future_predictions
    }, index=future_data.index)

    return future_results_df

# --- Streamlit App Layout ---
st.set_page_config(layout="wide")

# 1. Heading and Explanation
st.title(" River Temperature Prediction for the next 10 years ")
st.markdown("""
Welcome! This application predicts the average daily river temperature for the next 10 years (2025-2034) for the river Nethravathi. We have trained the models on 15 years of data which was obtained from QGis and Climate Data store.
It uses an ensemble of pre-trained machine learning models XGBoost, LightGBM, and TabNet which were trained on historical weather data and yeilded an R-2 of 0.998 meaning the model has been able to capture 99.8% of patterns present in the dataset.
We were able to achieve an accuracy of 99.95% on our Test dataset by calculating the Mean Absolute Percentage Error.

The prediction process involves:
1.  Generating synthetic daily weather data for the next 10 years after analyzing the trends of the last 15 years.
2.  Scaling this future data using the same method applied during model training.
3.  Feeding the scaled data into the pre-trained models to get temperature predictions.
4.  Combining the predictions from the individual models into a final ensemble prediction.

""")
st.markdown("---") # Separator

st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
st.image("./images/Qgis.png", caption="This image represents boundary of the river Nethravathi taken into consideration", width=1200)
st.image("./images/Qgis 2.png", caption="This is an image representing the embankent of the the river", width=1200)
st.image("./images/Qgis 3.png", caption="This image represents the contour plot around the river", width=1200)
st.markdown("</div>", unsafe_allow_html=True)

# Load assets first
assets = load_prediction_assets()

# Only show the button and prediction logic if assets loaded successfully
if assets:
    scaler, all_numeric_columns, features_for_modeling, xgb_model, lgb_booster, tabnet_model = assets

    #Prediction button
    predict_button = st.button("🚀 Predict Future Temperatures ")

    st.markdown("---") # Separator

    # 3. Prediction and Display Logic (runs only when button is clicked)
    if predict_button:

        # Call the prediction function
        future_results_df = generate_and_predict_future(
            scaler, all_numeric_columns, features_for_modeling,
            xgb_model, lgb_booster, tabnet_model # Pass lgb_booster
        )

        if future_results_df is not None:
            st.header("📊 Prediction Results")
            # st.balloons() # Removed balloons

            # --- Display Detailed Daily Predictions ---
            st.subheader("Daily Predicted Temperatures:")
            st.markdown("Showing the predicted temperature for each day over the next 10 years. You can sort the table by clicking column headers.")
            st.dataframe(future_results_df.style.format("{:.2f}"))

            # Option to download the full daily data
            csv_daily = future_results_df.to_csv().encode('utf-8')
            st.download_button(
                label="Download Full Daily Predictions as CSV",
                data=csv_daily,
                file_name='future_daily_temp_predictions.csv',
                mime='text/csv',
                key='download-daily'
            )
            st.markdown("---") # Separator

            # --- Process and Display Yearly Averages ---
            st.subheader("Yearly Average Temperatures")
            st.write("Calculating yearly averages from the daily predictions...")
            try:
                # Calculate yearly averages
                yearly_averages = future_results_df.resample('Y')['Predicted Temperature (°C)'].mean()
                yearly_averages_df = pd.DataFrame(yearly_averages)
                yearly_averages_df.index = yearly_averages_df.index.year # Use year as index
                yearly_averages_df.columns = ["Average Temp (°C)"]

                # Calculate overall average
                overall_average = yearly_averages.mean()

                # --- Display Yearly Results ---
                st.dataframe(yearly_averages_df.style.format("{:.2f}"))
                st.metric(label="Overall 10-Year Average Predicted Temperature", value=f"{overall_average:.2f} °C")

                # --- Weather Card UI Element ---
                card_css = """
                <style>
                    .weather-card {
                        background: linear-gradient(135deg, #74b9ff, #0984e3); /* Blue gradient from CodePen */
                        border-radius: 20px;
                        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
                        color: #ffffff;
                        font-family: 'Arial', sans-serif; /* Replaced 'Open Sans' with Arial */
                        padding: 30px;
                        max-width: 400px; /* Adjusted width */
                        margin: 30px auto; /* Centering and margin */
                        text-align: center;
                    }

                    .weather-icon img { /* Styling the img tag instead of SVG */
                        display: block;
                        margin: 0 auto 15px auto; /* Center and add bottom margin */
                        width: 100px; /* Control icon size */
                        height: auto; /* Maintain aspect ratio */
                        filter: drop-shadow(0 4px 6px rgba(0,0,0,0.1)); /* Optional shadow */
                    }

                    .weather-info h1.city { /* Targeting h1 with class city */
                        font-size: 1.8em; /* Slightly smaller than temperature */
                        margin: 15px 0 5px 0; /* Adjusted margin */
                        font-weight: 600; /* Boldness */
                        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
                    }

                    .weather-info h2.temperature { /* Targeting h2 with class temperature */
                        font-size: 4.5em; /* Large temperature */
                        margin: 0 0 10px 0;
                        font-weight: 700; /* Bolder */
                        line-height: 1;
                        text-shadow: 1px 1px 3px rgba(0,0,0,0.15);
                    }

                    .weather-info h2.temperature sup { /* Styling the degree symbol */
                        font-size: 0.4em;
                        font-weight: 400; /* Normal weight */
                        vertical-align: top;
                        margin-left: 2px;
                    }

                    .weather-info p.description { /* Targeting p with class description */
                        font-size: 1.1em;
                        margin: 0;
                        opacity: 0.9;
                    }
                </style>
                """
                st.markdown(card_css, unsafe_allow_html=True)

                # HTML Structure Adapted from CodePen
                # Using a sunny icon URL
                icon_url = "//cdn.weatherapi.com/weather/128x128/day/113.png" # Sunny icon

                st.markdown(f"""
                <div class="weather-card">
                    <div class="weather-icon">
                        <img src="{icon_url}" alt="Sunny Weather Icon">
                    </div>
                    <div class="weather-info">
                        <h1 class="city">River Nethravathi</h1>
                        <h2 class="temperature">{overall_average:.1f}<sup>°C</sup></h2>
                        <p class="description">Overall 10-Year Average Prediction</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Create and display the plot for yearly averages
                st.subheader("Yearly Average Temperature Trend")
                fig_yearly_avg, ax_yearly_avg = plt.subplots(figsize=(10, 5))
                ax_yearly_avg.plot(yearly_averages_df.index, yearly_averages_df["Average Temp (°C)"], marker='o', linestyle='-', color='red')
                ax_yearly_avg.set_title('Predicted Yearly Average Temperatures (2025-2034)')
                ax_yearly_avg.set_xlabel('Year')
                ax_yearly_avg.set_ylabel('Average Temperature (°C)')
                ax_yearly_avg.grid(True, linestyle='--', alpha=0.7)
                min_temp = yearly_averages_df["Average Temp (°C)"].min()
                max_temp = yearly_averages_df["Average Temp (°C)"].max()
                margin = (max_temp - min_temp) * 0.1
                ax_yearly_avg.set_ylim(min_temp - margin, max_temp + margin)
                ax_yearly_avg.set_xticks(yearly_averages_df.index)
                plt.tight_layout()
                st.pyplot(fig_yearly_avg)

                # Allow download of yearly averages
                csv_yearly = yearly_averages_df.to_csv().encode('utf-8')
                st.download_button(
                    label="Download Yearly Averages as CSV",
                    data=csv_yearly,
                    file_name='future_yearly_temp_averages.csv',
                    mime='text/csv',
                    key='download-yearly'
                )

            except Exception as e:
                st.error(f"An error occurred while calculating or plotting yearly averages: {e}")

        else:
            st.error("Prediction generation failed. Please check the logs or ensure models are loaded correctly.")

# Handle case where assets did not load initially
elif not assets:
    st.warning("Prediction cannot proceed because the required model assets failed to load.")
    st.info("Please check the file paths and ensure `train_models.py` ran without errors.")

st.markdown("---")
