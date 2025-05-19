import os
import streamlit as st
import requests
import numpy as np
import pandas as pd
from tensorflow.keras.initializers import Orthogonal
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import plotly.graph_objects as go

# --- Constants ---
SEQ_LENGTH = 30
PRED_LENGTH = 60

# --- Set page config - MUST be first Streamlit command ---
st.set_page_config(page_title='AQI Prediction Dashboard', page_icon='🌆', layout='wide')

# --- Load dataset and preprocess globally ---
@st.cache_data(show_spinner=False)
def load_and_prepare_data():
    df = pd.read_excel('WEEK 7 & 8/combined_AQI.xlsx')
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['weekday'] = df['date'].dt.day_name()
    df['year'] = df['date'].dt.year
    df = df.dropna(subset=['aqi']).reset_index(drop=True)

    # Encode city and weekday (fit encoders)
    city_encoder = OneHotEncoder(sparse_output=False)
    city_oh = city_encoder.fit_transform(df['city'].values.reshape(-1,1))

    weekday_encoder = OneHotEncoder(sparse_output=False)
    weekday_oh = weekday_encoder.fit_transform(df['weekday'].values.reshape(-1,1))

    # Prepare features for scaling AQI
    features = np.concatenate([
        df['aqi'].values.reshape(-1,1),
        city_oh,
        weekday_oh
    ], axis=1)

    scaler = MinMaxScaler(feature_range=(0,1))
    features[:,0:1] = scaler.fit_transform(features[:,0:1])

    return df, city_encoder, weekday_encoder, scaler

df, city_encoder, weekday_encoder, scaler = load_and_prepare_data()

# --- Load trained model ---
model = load_model('aqi_lstm_model.keras', custom_objects={'Orthogonal': Orthogonal})


# --- Weather API config ---
API_KEY = os.getenv('OPENWEATHER_API_KEY') or '79bf52094041ceda27f136d5dcd656f5'

# --- Weather fetching function ---
def get_weather(city, api_key):
    url = f'http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return {
            'temp': data['main']['temp'],
            'humidity': data['main']['humidity'],
            'description': data['weather'][0]['description'].title(),
            'wind_speed': data['wind']['speed']
        }
    return None

# --- AQI Prediction ---
def predict_aqi(city_name):
    city_name = city_name.title()
    city_data = df[df['city'] == city_name].copy()

    if len(city_data) < SEQ_LENGTH:
        return None, f"Not enough data for {city_name}. Need at least {SEQ_LENGTH} days."

    last_data = city_data.tail(SEQ_LENGTH)

    # Encode city and weekday
    city_oh = city_encoder.transform(last_data['city'].values.reshape(-1,1))
    weekday_oh = weekday_encoder.transform(last_data['weekday'].values.reshape(-1,1))

    features_arr = np.concatenate([
        last_data['aqi'].values.reshape(-1,1),
        city_oh,
        weekday_oh
    ], axis=1)

    features_arr[:, 0:1] = scaler.transform(features_arr[:, 0:1])

    input_data = features_arr.reshape(1, SEQ_LENGTH, features_arr.shape[1])

    prediction = model.predict(input_data)

    predicted_aqi = scaler.inverse_transform(prediction[0])

    return predicted_aqi.flatten(), None

# --- Aggregate predictions based on granularity ---
def aggregate_predictions(predictions, freq):
    if freq == 'Daily':
        return predictions
    elif freq == 'Weekly':
        weeks = len(predictions) // 7
        return [np.mean(predictions[i*7:(i+1)*7]) for i in range(weeks)]
    elif freq == 'Monthly':
        months = len(predictions) // 30
        return [np.mean(predictions[i*30:(i+1)*30]) for i in range(months)]
    else:
        return predictions

# --- Plot AQI gauge using Plotly ---
def plot_aqi_gauge(aqi_value, city):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=aqi_value,
        title={'text': f"Avg AQI for {city}"},
        gauge={
            'axis': {'range': [0, 500]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': 'green'},
                {'range': [51, 100], 'color': 'yellow'},
                {'range': [101, 150], 'color': 'orange'},
                {'range': [151, 200], 'color': 'red'},
                {'range': [201, 300], 'color': 'purple'},
                {'range': [301, 500], 'color': 'maroon'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': aqi_value
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

# --- Plotting function for matplotlib chart ---
def plot_aqi(city, values):
    plt.figure(figsize=(10, 4))
    plt.plot(values, marker='o', linestyle='-', color='blue', label='Predicted AQI')
    plt.title(f'AQI Forecast for {city}', fontsize=16)
    plt.xlabel('Days Ahead', fontsize=12)
    plt.ylabel('AQI', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf

# --- Streamlit UI ---

# Centered Title & Description
st.markdown("<h1 style='text-align: center;'>🌆 AQI Prediction Dashboard with Weather Info</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:16px; color:gray;'>Welcome! Enter a city name in the sidebar to view the <b>current weather</b> and <b>AQI forecast</b> for the next 60 days.</p>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar Input
city_input = st.sidebar.text_input('Enter City Name').strip().title()

# Main content container
with st.container():
    if city_input:
        # Create two main columns: Weather info and AQI info side by side
        col1, col2 = st.columns([1, 2], gap="large")

        # Weather Info
        with col1:
            weather = get_weather(city_input, API_KEY)
            if weather:
                st.markdown(f"### Current Weather in {city_input}")
                st.metric('Temperature (°C)', f"{weather['temp']} °C")
                st.metric('Humidity (%)', f"{weather['humidity']}%")
                st.write(f"**Condition:** {weather['description']}")
                st.write(f"**Wind Speed:** {weather['wind_speed']} m/s")
            else:
                st.error('Weather data not available for this city.')

        # AQI Prediction & Plots with selector and gauge side-by-side
        with col2:
            # Two columns inside col2 for gauge and selector horizontally aligned
            gauge_col, selector_col = st.columns([2, 1])

            with selector_col:
                freq = st.selectbox("Select Forecast Granularity:", options=['Daily', 'Weekly', 'Monthly'])

            if st.button('Predict AQI'):
                predicted_aqi, error_msg = predict_aqi(city_input)
                if error_msg:
                    st.error(error_msg)
                else:
                    st.success(f'Predicted AQI for {city_input} ({freq} forecast)')

                    # Aggregate AQI based on selected frequency for gauge
                    agg_pred = aggregate_predictions(predicted_aqi, freq)
                    avg_aqi = np.mean(agg_pred)

                    with gauge_col:
                        st.plotly_chart(plot_aqi_gauge(avg_aqi, city_input), use_container_width=True)

                    # Line chart full width below
                    st.line_chart(predicted_aqi)

                    # Matplotlib image below line chart
                    st.image(plot_aqi(city_input, predicted_aqi), caption=f'AQI Forecast for {city_input}')
    else:
        st.info("Please enter a city name in the sidebar to get started.")

# Footer
st.markdown("---")
st.markdown("<center>Developed by You | 2025</center>", unsafe_allow_html=True)
