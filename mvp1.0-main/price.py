import re
import numpy as np
import pickle
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import Ollama

# Initialize Ollama model
llm = Ollama(model="llama2-uncensored")

# Function to load the XGBoost model from a file
def load_xgboost_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Function to parse details from the user's query
def parse_details_from_query(query):
    details = {
        "material_type": None,
        "square_feet": None,
        "location": None
    }

    # Extract material type
    material_match = re.search(r'\b(bricks|lumber|cabinets|roofing|countertops|steel|concrete|stone|finishes|windows|flooring|wood)\b', query, re.IGNORECASE)
    if material_match:
        details["material_type"] = material_match.group(0).lower()

    # Extract square footage
    sqft_match = re.search(r'(\d+(?:,\d+)*)\s*sqft', query, re.IGNORECASE)
    if sqft_match:
        details["square_feet"] = int(sqft_match.group(1).replace(",", ""))

    # Extract location
    location_match = re.search(r'\b\w+,\s+\w+\b', query)  # Simple regex for city, state format
    if location_match:
        details["location"] = location_match.group(0)

    return details

# Function to get price estimation based on parsed details
def get_price_estimation(details, model):
    # Assuming your model expects material type, square footage, and location as input features
    # Dummy encoding for material type as an example
    material_types = ['bricks', 'lumber', 'cabinets', 'roofing', 'countertops', 'steel', 'concrete', 'stone', 'finishes', 'windows', 'flooring', 'wood']
    material_vector = [1 if details["material_type"] == material else 0 for material in material_types]

    input_features = np.array(material_vector + [details["square_feet"]]).reshape(1, -1)

    # Predict the price
    predicted_price = model.predict(input_features)
    return predicted_price[0]

# Interactive chat function
def chat():
    print("Hello my name is Chris your guide to Cost Predict! I'm here to help you with real-time cost estimation for building materials.")
    model = load_xgboost_model('xgboost_model.pkl')

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        # Use Ollama for additional processing or generating a response
        ollama_response = llm(user_input)

        # Logic to parse and respond to user queries
        details = parse_details_from_query(user_input)
        if all(value is not None for value in details.values()):
            price_estimate = get_price_estimation(details, model)
            print(f"CostPredict: The estimated cost for {details['material_type']} is approximately ${price_estimate:.2f} for {details['square_feet']} sqft in {details['location']}.")
        else:
            # Use Ollama's response or ask for more details
            print(f"CostPredict: {ollama_response}")

if __name__ == "__main__":
    chat()