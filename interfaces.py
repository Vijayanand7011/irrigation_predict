

from flask import Flask, render_template, request, redirect, url_for
from datetime import datetime
import pandas as pd
import os
import csv



################################################################################################################################################################
# CSV FILE PATH -> OS DIRECT PATH
csv_file_pro = os.path.join(os.path.dirname(__file__), 'status_data.csv')
csv_file_soil_data = os.path.join(os.path.dirname(__file__), 'soil_data.csv')

print(csv_file_pro)
print(csv_file_soil_data)
################################################################################################################################################################


################################################################################################################################################################
# --------------------------------------------------------------------------------------------------------------------------------------------------------------
# PARAMETERS.PY



import requests
import csv
from datetime import datetime
import math


def calculate_solar_radiation(coordinates_dict):
    """
    Calculates solar radiation for a given set of coordinates (longitude, latitude) on the current date.

    Parameters:
    - coordinates_dict: A list of dictionaries where each dictionary has 'lon' for longitude and 'lat' for latitude.

    Returns:
    - A list of solar radiation values in MJ/m², rounded to two decimal places.
    """

    # Solar constant (W/m²)
    solar_constant = 1361  # W/m²

    # Convert solar constant to MJ/m² (1 W/m² = 1e-3 MJ/m²)
    solar_constant_mj = solar_constant * 1e-3

    # Get the current date
    date_obj = datetime.today()
    n = date_obj.timetuple().tm_yday  # Day of the year

    # Results list to store the solar radiation values
    solar_radiation_results = []

    # Loop through all coordinates in the input dictionary
    for coordinates in coordinates_dict:
        # Extract latitude and longitude
        latitude = coordinates['lat']
        longitude = coordinates['lon']

        # Latitude and longitude in radians
        latitude_rad = math.radians(latitude)

        # Calculate solar declination (δ) in radians
        declination = 23.44 * math.sin(math.radians((360 / 365) * (n - 81)))

        # Calculate the solar declination in radians
        declination_rad = math.radians(declination)

        # Solar hour angle (for noon, this is 0°)
        # We assume solar noon for simplicity, so hour angle H = 0.
        hour_angle = 0

        # Calculate solar elevation angle (α)
        sin_alpha = math.sin(declination_rad) * math.sin(latitude_rad) + \
                    math.cos(declination_rad) * math.cos(latitude_rad) * math.cos(math.radians(hour_angle))

        # Ensure the value is within the range [-1, 1] to avoid math errors
        sin_alpha = max(-1, min(1, sin_alpha))

        # Solar elevation angle (α)
        alpha = math.asin(sin_alpha)

        # Calculate solar radiation (I) in W/m²
        # Equation for solar radiation
        solar_radiation_wm2 = solar_constant * (1 + 0.034 * math.cos(2 * math.pi / 365 * n)) / math.cos(alpha)

        # Convert the solar radiation to MJ/m²
        solar_radiation_mj = solar_radiation_wm2 * 1e-3  # Convert W/m² to MJ/m²

        # Append the result rounded to two decimal places
        solar_radiation_results.append(round(solar_radiation_mj, 2))

    return solar_radiation_results[0]




def get_soil_data(pincode, filename):
    """
    Function to read a CSV file and return soil data based on a given pincode.

    :param pincode: The pincode to search for (as string).
    :param filename: The name of the CSV file containing the data.
    :return: A dictionary containing the soil data for the matching record or None if no match is found.
    """
    # Convert the input pincode to integer
    pincode = int(pincode)

    with open(filename, mode='r') as file:
        reader = csv.DictReader(file)

        # Print out the column names for debugging
        print(f"CSV Headers: {reader.fieldnames}")

        for row in reader:
            # Check if the required columns exist
            if 'pincode_start' in row and 'pincode_end' in row:
                pincode_start = int(row['pincode_start'])
                pincode_end = int(row['pincode_end'])

                # Check if the pincode falls within the range
                if pincode_start <= pincode <= pincode_end:
                    # Return the data from the matching row (excluding pincode fields)
                    soil_data = {
                        'soil_type': row['soil_type'],
                        'drainage': row['drainage'],
                        'infiltration': row['infiltration'],
                        'wilting_point': row['wilting_point'],
                        'field_capacity': row['field_capacity']
                    }
                    return soil_data
            else:
                print("Error: Missing expected columns 'pincode_start' or 'pincode_end' in the row.")

    # If no matching record is found
    return None


# Function to fetch weather data by pincode
def get_weather_data(pincode, api_key):
    """
    Function to fetch weather data based on the given pincode from OpenWeatherMap API.

    :param pincode: The pincode to get the weather for.
    :param api_key: The API key for OpenWeatherMap.
    :return: A dictionary containing weather data or None if the request fails.
    """
    # OpenWeatherMap API endpoint for weather data by zip code
    url = f'http://api.openweathermap.org/data/2.5/weather?zip={pincode},IN&appid={api_key}&units=metric'

    # Send a GET request to fetch the weather data
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()
        print(data)

        # Extract necessary data from the response
        weather_data = {
            'coordinates': data['coord'],
            'country': data['sys']['country'],
            'city': data['name'],
            'temp': data['main']['temp'],
            'temp_max': data['main']['temp_max'],
            'temp_min': data['main']['temp_min'],
            'humidity': data['main']['humidity'],
            'wind_speed': data['wind']['speed'],
            'weather_description': data['weather'][0]['description'],
            'solar_radiation': calculate_solar_radiation([data['coord']])*10,
            'pressure': data['main']['pressure'] / 1000,

        }

        # Return weather data
        return weather_data
    else:
        # If the request failed, return None
        return None


# Main function to get user input and fetch both soil and weather data
def get_field_data(pincode):
    # Example API key (replace with your own key)
    weather_api_key = 'bd5e378503939ddaee76f12ad7a97608'

    # Fetch soil and weather data
    soil_data = get_soil_data(pincode, csv_file_soil_data)
    weather_data = get_weather_data(pincode, weather_api_key)

    if soil_data is None:
        print("No soil data found for the given pincode.")
    if weather_data is None:
        print("No weather data found for the given pincode.")

    if soil_data is not None and weather_data is not None:
        # Merge the data only if both are available
        field_data = {**weather_data, **soil_data}
        print(field_data)
        return field_data
    else:
        print("Missing data for the given pincode.")
        return None


#--------------------------------------------------------------------------------------------------------------------------------------------------------------
################################################################################################################################################################



################################################################################################################################################################
# --------------------------------------------------------------------------------------------------------------------------------------------------------------
# Importing Machine Laerning Module to fetch predicted  .PY

import joblib
import pandas as pd
from ML_model import predict_irrigation_requirement

#--------------------------------------------------------------------------------------------------------------------------------------------------------------
################################################################################################################################################################




################################################################################################################################################################
# --------------------------------------------------------------------------------------------------------------------------------------------------------------
# file .PY




#--------------------------------------------------------------------------------------------------------------------------------------------------------------
################################################################################################################################################################




# Function to run the Flask app
def run_flask_app():

    app_interface = Flask(__name__)

    # Check if the CSV file exists, if not, create it with headers
    if not os.path.exists(csv_file_pro):
        with open(csv_file_pro, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                'crop','pincode', 'timestamp', 'upper_temp', 'lower_temp', 'humidity', 'rainfall',
                'wind_speed', 'solar_rad', 'soil_type', 'soil_moisture', 'infiltration_rate',
                'field_capacity', 'wilting_point', 'growth_stage', 'root_depth', 'crop_coeff',
                'irrigation_efficiency', 'water_quality', 'slope', 'drainage_conditions'
            ])

    # -------------------------------------------------------------------------------------------------------------------------------------------------------

    @app_interface.route('/')
    def home():
        return render_template('home.html')

    # -------------------------------------------------------------------------------------------------------------------------------------------------------
    @app_interface.route('/references')
    def references():
        return render_template('references.html')

    # -------------------------------------------------------------------------------------------------------------------------------------------------------
    @app_interface.route('/references/<crop>')
    def crop_specific_reference(crop):
        file_name = f"references/{crop}"
        return render_template(file_name)

    # -------------------------------------------------------------------------------------------------------------------------------------------------------
    @app_interface.route('/tutorial/how-much-to-irrigate')
    def irrigation_factor_tutorial():
        return render_template('irrigation_factor_tutorial.html')

    # -------------------------------------------------------------------------------------------------------------------------------------------------------
    @app_interface.route('/tutorial/pressure-compensation-drip-irrigation')
    def pressure_compensation_drip_irriagtion_tutorial():
        return render_template('pressure_compensation_drip_irrigation.html')

    # -------------------------------------------------------------------------------------------------------------------------------------------------------
    @app_interface.route('/tutorial/subsurface-drip-irrigation')
    def subsurface_drip_irriagtion_tutorial():
        return render_template('subsurface_drip_irrigation.html')

    # -------------------------------------------------------------------------------------------------------------------------------------------------------

    #-------------------------------------------------------------------------------------------------------------------------------------------------------
    # <<<----------- PRO ----------->>>
    # Route to handle the form input
    @app_interface.route('/input-pro', methods=['GET', 'POST'])
    def input_pro():  # renamed function
        if request.method == 'POST':
            # Collecting input data from the form
            crop = request.form['crop']
            pincode = int(request.form['pincode'])
            timestamp = datetime.utcnow()
            upper_temp = float(request.form['upper_temp'])
            lower_temp = float(request.form['lower_temp'])
            humidity = float(request.form['humidity'])
            wind_speed = float(request.form['wind_speed'])
            rainfall = float(request.form['rainfall'])
            solar_rad = float(request.form['solar_rad'])
            soil_type = request.form['soil_type']
            soil_moisture = float(request.form['soil_moisture'])
            infiltration_rate = float(request.form['infiltration_rate'])
            field_capacity = float(request.form['field_capacity'])
            wilting_point = float(request.form['wilting_point'])
            growth_stage = request.form['growth_stage']
            root_depth =float( request.form['root_depth'])
            crop_coeff = float(request.form['crop_coeff'])
            irrigation_efficiency = float(request.form['irrigation_efficiency'])
            water_quality = request.form['water_quality']
            water_ava = request.form['water_ava']
            slope = float(request.form['slope'])
            drainage_conditions = request.form['drainage_conditions']

            model_data = {
                'Upper_Temperature': upper_temp,
                'Lower_Temperature': lower_temp,
                'Humidity': humidity,
                'Rainfall': rainfall,
                'Wind Speed': wind_speed,
                'Solar Radiation': solar_rad,
                'Soil Type': soil_type,
                'Soil Moisture Content': soil_moisture,
                'Infiltration Rate': infiltration_rate,
                'Field Capacity': field_capacity,
                'Wilting Point': wilting_point,
                'Growth Stage': growth_stage,
                'Root Depth': root_depth,
                'Crop Coefficient': crop_coeff,
                'Water Availability':water_ava,
                'Irrigation System Efficiency': irrigation_efficiency,
                'Water Quality': water_quality,
                'Slope': slope,
                'Drainage Conditions': drainage_conditions,
            }
            type = 'PRO'

            water_supplied = predict_irrigation_requirement(model_data, crop)

            # Prepare the row to be appended to the CSV
            new_row = [
                crop,pincode, timestamp, upper_temp, lower_temp, humidity, rainfall,
                wind_speed, solar_rad, soil_type, soil_moisture, infiltration_rate,
                field_capacity, wilting_point, growth_stage, root_depth, crop_coeff,
                irrigation_efficiency, water_quality, slope, drainage_conditions, water_supplied,type
            ]

            # Append the data to the CSV file
            try:
                with open(csv_file_pro, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(new_row)
                print("Data successfully appended to the CSV!--PRO")
            except Exception as e:
                print(f"Error while writing to CSV --PRO-- : {e}")
                return render_template('input_pro.html', error="Error processing the data.")

            return redirect(url_for('output_pro'))

        return render_template('input_pro.html')

    #---------------------------------------------------------------------------------------------------------------------------------------------------------
    # <<<----------- PRO ----------->>>
    # Function to fetch data from the CSV and generate HTML
    def css_data_fetch_pro(csv_path_pro):

        # Check if the file exists
        if not os.path.exists(csv_file_pro):
            return f"Error: CSV file '{csv_path}' does not exist."

        # Read the CSV file using pandas
        try:
            df = pd.read_csv(csv_file_pro)
        except Exception as e:
            return f"Error reading CSV file: {str(e)}"

        # Check if there's data in the CSV file
        if df.empty:
            return "Error: CSV file is empty."

        # Get the last record from the CSV
        last_record = df.iloc[-1]  # Last row

        # Prepare data to display in HTML
        data = {
            "title": "Predicted Output",
            "description": "On Basis Of The Data You Entered The Required Supply Of Water For The Crops Has Been Predicted",
            "items": [
                f"Crop: {last_record['crop']}",
                f"Pincode: {last_record['pincode']}",
                f"Timestamp: {last_record['timestamp']}",
                f"Upper Temperature: {last_record['upper_temp']}",
                f"Lower Temperature: {last_record['lower_temp']}",
                f"Humidity: {last_record['humidity']}",
                f"Rainfall: {last_record['rainfall']}",
                f"Wind Speed: {last_record['wind_speed']}",
                f"Solar Radiation: {last_record['solar_rad']}",
                f"Soil Type: {last_record['soil_type']}",
                f"Soil Moisture: {last_record['soil_moisture']}",
                f"Infiltration Rate: {last_record['infiltration_rate']}",
                f"Field Capacity: {last_record['field_capacity']}",
                f"Wilting Point: {last_record['wilting_point']}",
                f"Growth Stage: {last_record['growth_stage']}",
                f"Root Depth: {last_record['root_depth']}",
                f"Crop Coefficient: {last_record['crop_coeff']}",
                f"Irrigation Efficiency: {last_record['irrigation_efficiency']}",
                f"Water Quality: {last_record['water_quality']}",
                f"Slope: {last_record['slope']}",
                f"Drainage Conditions: {last_record['drainage_conditions']}",
                f"Water Require For Crop: {last_record['water_supplied']}",
                f"Input Type: {last_record['type']}"
            ]
        }

        # Return the rendered HTML with data
        return render_template('output_pro.html', **data)

    # Route to handle the dynamic path and display CSV content
    @app_interface.route('/output-pro')
    def output_pro():
        try:
            return css_data_fetch_pro(csv_file_pro)
        except Exception as e:
            return f"Error occurred: {str(e)}"

    # -------------------------------------------------------------------------------------------------------------------------------------------------------
    # <<<----------- PIN ----------->>>
    # Route to handle the form input    <CROP>
    @app_interface.route('/input-pin', methods=['GET', 'POST'])
    def input_pin():  # renamed function
        pincode_data = None
        if request.method == 'POST':
            # Collecting input data from the form
            crop = request.form['crop']
            pincode = int(request.form['pincode'])
            timestamp = datetime.utcnow()
            rainfall = float(request.form['rainfall'])
            soil_type = request.form['soil_type']
            soil_moisture = float(request.form['soil_moisture'])
            growth_stage = request.form['growth_stage']
            root_depth = float(request.form['root_depth'])
            crop_coeff = float(request.form['crop_coeff'])
            irrigation_efficiency = float(request.form['irrigation_efficiency'])
            water_quality = request.form['water_quality']
            slope = float(request.form['slope'])

            pincode_data = get_field_data(pincode)
            # Check if pincode_data is None
            if pincode_data is None:
                return render_template('input_pin.html', error="No data found for the given pincode.")

            model_data = {
                'Upper_Temperature': pincode_data['temp_max'],
                'Lower_Temperature': pincode_data['temp_min'],
                'Humidity': pincode_data['humidity'],
                'Rainfall': rainfall,
                'Wind Speed': pincode_data['wind_speed'],
                'Solar Radiation': pincode_data['solar_radiation'],
                'Soil Type': soil_type,
                'Soil Moisture Content': soil_moisture,
                'Infiltration Rate': pincode_data['infiltration'],
                'Field Capacity': pincode_data['field_capacity'],
                'Wilting Point': pincode_data['wilting_point'],
                'Growth Stage': growth_stage,
                'Root Depth': root_depth,
                'Crop Coefficient': crop_coeff,
                'Irrigation System Efficiency': irrigation_efficiency,
                'Water Quality': water_quality,
                'Slope': slope,
                'Drainage Conditions': pincode_data['drainage']
            }
            type = 'PIN'

            water_supplied = predict_irrigation_requirement(model_data, crop)
            # Prepare the row to be appended to the CSV
            new_row = [
                crop,pincode, timestamp, pincode_data['temp_max'], pincode_data['temp_min'], pincode_data['humidity'],
                rainfall,
                pincode_data['wind_speed'], pincode_data['solar_radiation'], soil_type,
                soil_moisture, pincode_data['infiltration'], pincode_data['field_capacity'],
                pincode_data['wilting_point'],
                growth_stage, root_depth, crop_coeff, irrigation_efficiency, water_quality, slope,
                pincode_data['drainage'], water_supplied, type
            ]

            # Append the data to the CSV file
            try:
                with open(csv_file_pro, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(new_row)
                print("Data successfully appended to the CSV!-- PIN")
            except Exception as e:
                print(f"Error while writing to CSV --PIN-- : {e}")
                return render_template('input_pin.html', error="Error processing the data.")

            return redirect(url_for('output_pro'))
        if request.method == 'POST':
            print(request.form)

        return render_template('input_pin.html')


    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------

    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------

    # <<!----------!__________!~~~~~~~~~~!`````````` Final Flask App Return ``````````!~~~~~~~~~~!__________!----------!>>
    print(app_interface.url_map)
    return app_interface    #  return app_interface = Flask(__name__) then put app = run_flask_app; then enter app.run(debug=True, port=5000 )
