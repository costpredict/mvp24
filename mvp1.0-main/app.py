import re
import numpy as np
import pickle
from datetime import datetime
import gradio as gr

# Load the XGBoost model
def load_xgboost_model(model_path='xgboost_model.pkl'):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

model = load_xgboost_model()

# Function to extract details from the query
def parse_details_from_query(query):
    # Regex for extracting material type and square footage
    material_match = re.search(r'\b(bricks|lumber|cabinets|roofing|countertops|steel|concrete|stone|finishes|windows|flooring|wood)\b', query, re.IGNORECASE)
    sqft_match = re.search(r'(\d+)\s*sqft', query, re.IGNORECASE)
    # Regex for extracting dates in specified formats
    date_match = re.search(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}|\b\d{1,2}-\d{1,2}-\d{4}', query)

    material = material_match.group(0).lower() if material_match else None
    square_feet = int(sqft_match.group(1)) if sqft_match else None
    date = datetime.strptime(date_match.group(0), '%B %d, %Y') if date_match else None
    if not date:
        date = datetime.strptime(date_match.group(0), '%m-%d-%Y') if date_match else None
    
    return material, square_feet, date

# Convert parsed details to model features
def details_to_model_features(material, square_feet, date):
    # Assuming material is one-hot encoded and date is decomposed into year and month
    # This is a placeholder; replace with your actual feature engineering logic
    features = np.array([square_feet, date.year, date.month])
    return features

# Function for price prediction
def predict_price(query):
    material, square_feet, date = parse_details_from_query(query)
    if not all([material, square_feet, date]):
        return "Missing information. Please include material type, square footage, and date."
    
    features = details_to_model_features(material, square_feet, date)
    prediction = model.predict([features])[0]
    return f"Estimated price for {material} is ${prediction:.2f}."

# Gradio interface
iface = gr.Interface(fn=predict_price, inputs="text", outputs="text", title="Cost Predict", description="Ask about material costs. Include material type, square footage, and date.")
iface.launch()