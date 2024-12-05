import joblib
import pandas as pd  # Correct import for pandas


# Function to predict irrigation requirement based on input data from a dictionary
def predict_irrigation_requirement(input_data: dict, crop: str):
    """
    Predict irrigation requirement based on input data from a dictionary and return the result.

    Args:
    input_data (dict): Dictionary containing the input features for prediction.

    Returns:
    float: Predicted irrigation requirement in mm.
    """
    # Construct the model filename using the crop name
    model_filename = f'model_joblib/{crop}.joblib'

    # Load the model dynamically
    model = joblib.load(model_filename)

    # Convert the input dictionary into a DataFrame
    input_df = pd.DataFrame([input_data])

    # Predict the irrigation requirement using the loaded model
    irrigation_requirement = model.predict(input_df)

    # Round of the requirement to two decimal places
    irrigation_requirement[0] = round(irrigation_requirement[0], 2)

    # Return the predicted irrigation requirement
    return irrigation_requirement[0]

"""
#------------------------------------------ Data Input Example ----------------------------------------------
input_data = {
            'Upper_Temperature': 35.0,  # Example values
            'Lower_Temperature': 22.0,
            'Humidity': 75.0,
            'Rainfall': 10.0,
            'Wind Speed': 2.0,
            'Solar Radiation': 20.0,
            'Soil Type': 'Loamy',
            'Soil Moisture Content': 20.0,
            'Infiltration Rate': 5.0,
            'Field Capacity': 30.0,
            'Wilting Point': 12.0,
            'Growth Stage': 'Vegetative',
            'Root Depth': 40.0,
            'Crop Coefficient': 1.2,
            'Water Availability':'Low',
            'Irrigation System Efficiency': 85.0,
            'Water Quality': 'Good',
            'Slope': 5.0,
            'Drainage Conditions': 'Well-drained'
        }
        

print(predict_irrigation_requirement(input_data,'wheat'))
"""
#------------------------------------------------------------------------------------------------------------------------------------