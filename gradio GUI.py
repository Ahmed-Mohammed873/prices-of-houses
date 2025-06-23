import gradio as gr
import joblib
import numpy as np

model = joblib.load("linear_model.pkl")

def predict_price(gr_liv_area, bedrooms, bathrooms):
    features = np.array([[gr_liv_area, bedrooms, bathrooms]])
    prediction = model.predict(features)[0]
    return f"🏷️ Estimated Price: ${prediction:,.0f}"

interface = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Number(label="🏠 Living Area (sq ft)", value=1500),
        gr.Number(label="🛏️ Bedrooms", value=3),
        gr.Number(label="🛁 Bathrooms", value=2),
    ],
    outputs=gr.Textbox(label="Predicted House Price"),
    title="House Price Predictor 🏡",
    description="Enter property details to get an estimated house price using a trained linear regression model."
)

interface.launch()
