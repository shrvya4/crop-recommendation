from flask import Flask, request, jsonify
import pickle  # For loading the saved ML model
import re

app = Flask(__name__)

# Load the pre-trained machine learning model at the start
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)
with open('ordinal_enc.pkl', 'rb') as ordinal_file:
    ordinal = pickle.load(ordinal_file)
with open('x_train.pkl', 'rb') as xtrain_file:
    X_train = pickle.load(xtrain_file)
with open('label_enc.pkl', 'rb') as label_file:
    label_encoder = pickle.load(label_file)

@app.route('/')
def home():
    return '''
  <!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Soil Quality Prediction</title>
    <style>
        /* Body background with teaser image */
        body {
            font-family: 'Verdana', sans-serif; /* Updated font for a cleaner look */
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0;
            height: 100vh;
            background-size: cover;
            color: #34495e; /* Text color for readability */
            transition: background-color 0.5s ease-in-out; /* Smooth transition for any changes */
        }

        .container {
            text-align: center;
            padding: 20px;
        }

        /* Card styling to stand out against the background */
        .card {
            background-color: rgba(255, 255, 255, 0.9); /* Slight transparency */
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            max-width: 400px;
            margin: 0 auto;
            padding: 20px;
            text-align: left;
        }

        h1 {
            color: #4CAF50; /* Green color */
            padding-top: 10%;
            font-family: 'Georgia', serif; /* Unique, classic font */
            font-weight: bold;
            letter-spacing: 2px; /* Adds space between letters for style */
            text-transform: uppercase; /* Makes text all uppercase */
            text-shadow: 1px 1px 5px rgba(0, 0, 0, 0.3); /* Adds a subtle shadow for depth */
}

        label {
            display: block;
            margin: 10px 0 5px;
            color: #34495e;
            font-weight: bold;
        }

        select, input[type="number"] {
            width: 100%;
            padding: 8px;
            margin: 5px 0 10px;
            border: 1px solid #d1d5db;
            border-radius: 4px;
        }

        button {
            width: 100%;
            padding: 10px;
            background-color: #4CAF50;
            border: none;
            color: white;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            border-radius: 4px;
            transition: background-color 0.3s;
        }

        button:hover {
            background-color: #45a049;
        }

        #result {
            margin-top: 15px;
            font-weight: bold;
            color: #e74c3c;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Soil to Success: Uncover the Best Crops for Your Field Conditions</h1>
        <p>Find Your Perfect Crop Match! Simply fill in the values and click Submit!</p>

        <div class="card">
            <form id="myForm">
                <label for="humidity">Humidity:</label>
                <select id="humidity">
                    <option value="Low">Low</option>
                    <option value="Moderate">Moderate</option>
                    <option value="High">High</option>
                    <option value="Very High">Very High</option>
                </select>

                <label for="zone">Zone:</label>
                <select id="zone">
                    <option value="13A">13A</option>
                    <option value="13B">13B</option>
                    <option value="12A">12A</option>
                    <option value="12B">12B</option>
                    <option value="11A">11A</option>
                    <option value="11B">11B</option>
                    <option value="10A">10A</option>
                    <option value="10B">10B</option>
                    <option value="9A">9A</option>
                    <option value="9B">9B</option>
                    <option value="8A">8A</option>
                    <option value="8B">8B</option>
                    <option value="7A">7A</option>
                    <option value="7B">7B</option>
                    <option value="6A">6A</option>
                    <option value="6B">6B</option>
                    <option value="5A">5A</option>
                    <option value="5B">5B</option>
                    <option value="4A">4A</option>
                    <option value="4B">4B</option>
                    <option value="2A">2A</option>
                    <option value="2B">2B</option>
                </select>

                <label for="temperatureStartRange">Temperature Start Range (°C):</label>
                <input type="number" id="temperatureStartRange" step="0.1" placeholder="e.g., 15.0">

                <label for="temperatureEndRange">Temperature End Range (°C):</label>
                <input type="number" id="temperatureEndRange" step="0.1" placeholder="e.g., 25.0">

                <label for="salinity">Salinity:</label>
                <select id="salinity">
                    <option value="Non-Saline">Non-Saline</option>
                    <option value="Slightly Saline">Slightly Saline</option>
                    <option value="Moderately Saline">Moderately Saline</option>
                    <option value="Highly Saline">Highly Saline</option>
                </select>

                <label for="organicMatterContent">Organic Matter Content:</label>
                <select id="organicMatterContent">
                    <option value="Low">Low</option>
                    <option value="Moderate">Moderate</option>
                    <option value="High">High</option>
                </select>

                <label for="soilType">Soil Type:</label>
                <select id="soilType">
                    <option value="Slightly Acidic">Slightly Acidic</option>
                    <option value="Neutral">Neutral</option>
                    <option value="Slightly Alkaline">Slightly Alkaline</option>
                </select>

                <label for="soilTexture">Soil Texture:</label>
                <select id="soilTexture">
                    <option value="Loamy Soil">Loamy Soil</option>
                    <option value="Silt Soil">Silt Soil</option>
                    <option value="Sandy Loam">Sandy Loam</option>
                </select>

                <label for="plantType">Plant Type:</label>
                <select id="planttype">
                    <option value="Vegetable">Vegetable</option>
                    <option value="Root">Root</option>
                    <option value="Leafy Green">Leafy Green</option>
                    <option value="Fruit">Fruit</option>
                    <option value="Legume">Legume</option>
                    <option value="Nut">Nut</option>
                    <option value="Grain">Grain</option>
                    <option value="Tubers">Tubers</option>
                    <option value="Herb">Herb</option>
                    <option value="Mushroom">Mushroom</option>
                </select>

                <button type="submit">Submit</button>
            </form>
            <div id="result"></div>
        </div>
    </div>

    <script>
        document.getElementById('myForm').addEventListener('submit', function(e) {
            e.preventDefault();

            var humidity = document.getElementById('humidity').value;
            var zone = document.getElementById('zone').value;
            var temperatureStartRange = document.getElementById('temperatureStartRange').value;
            var temperatureEndRange = document.getElementById('temperatureEndRange').value;
            var salinity = document.getElementById('salinity').value;
            var organicMatterContent = document.getElementById('organicMatterContent').value;
            var soilType = document.getElementById('soilType').value;
            var soilTexture = document.getElementById('soilTexture').value;
            var plantype = document.getElementById('planttype').value;

            var data = {
                humidity: humidity,
                zone: zone,
                temperature_start_range: temperatureStartRange,
                temperature_end_range: temperatureEndRange,
                salinity: salinity,
                organic_matter_content: organicMatterContent,
                soil_type: soilType,
                soil_texture: soilTexture,
                plant_type: plantype 
            };

            fetch('/run_model', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(data)
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('result').innerHTML = "Prediction: " + data.result;
            })
            .catch(error => console.error('Error:', error));
        });
    </script>
</body>
</html>


    '''

@app.route('/run_model', methods=['POST'])

def run_model():

    # Extract input data from the request
    humidity_str = str(request.json['humidity'])
    humidity_mapping = {'Low': 1, 'Moderate': 2, 'High': 3, 'Very High': 4}
    humidity_int = humidity_mapping[humidity_str]

    zone_str = str(request.json['zone'])
    zone_int = float(zone_str[:-1])  # Extract the numeric part

    temperature_start_range = float(request.json['temperature_start_range'])
    temperature_end_range = float(request.json['temperature_end_range'])

    salinity_str = str(request.json['salinity'])
    salinity_mapping = {'Non-Saline': 1, 'Slightly Saline': 2, 'Moderately Saline': 3, 'Strongly Saline': 4}
    salinity_int = salinity_mapping[salinity_str]


    organic_matter_content_str = str(request.json['organic_matter_content'])
    organic_matter_mapping = {'Low': 1, 'Moderate': 2, 'High': 3}
    organic_matter_content_int = organic_matter_mapping[organic_matter_content_str]

    soil_type_str =    str(request.json['soil_type'])
    soil_type_mapping = {'Slightly Acidic': 1, 'Neutral': 2, 'Slightly Alkaline': 3, 'Moderately Alkaline': 4}
    soil_type_int = soil_type_mapping[soil_type_str]

    soil_texture_str = str(request.json['soil_texture'])
    soil_texture_mapping = {'Sandy Soil': 1, 'Sandy Loam': 2, 'Loamy Soil': 3, 'Silt Soil': 4}
    soil_texture_int = soil_texture_mapping[soil_texture_str]

    plant_type_str =   str(request.json['plant_type'])
    plant_type_mapping = {'Fruit': 0, 'Grain': 1, 'Herb': 2, 'Leafy Green': 3, 'Legume': 4, 'Mushroom': 5, 'Nut': 6, 'Root': 7, 'Tubers' : 8, 'Vegetable': 9}
    plant_type_int = plant_type_mapping[plant_type_str]

    print(humidity_int, zone_int, temperature_start_range, temperature_end_range, salinity_int, organic_matter_content_int, soil_type_int, soil_texture_int, plant_type_int)
    print(humidity_str, zone_str, temperature_start_range, temperature_end_range, salinity_str, organic_matter_content_str, soil_type_str, soil_texture_str, plant_type_str)

    features = [[ plant_type_int, humidity_int, zone_int, temperature_start_range, temperature_end_range, salinity_int, organic_matter_content_int, soil_type_int, soil_texture_int]]
    scaled_features = scaler.transform(features)
    print(features)
    print(scaled_features)
    scaled_features_2 = scaled_features
    import numpy as np
    sample_input = np.array(scaled_features).reshape(1, -1) #Reshape the sample input into a 2D array with one row and the required number of columns
    
    
    # Explain the prediction for the sample input
    import lime
    from lime.lime_tabular import LimeTabularExplainer

    explainer = LimeTabularExplainer(X_train.values,
                                 feature_names=X_train.columns.tolist(),
                                 class_names=label_encoder.classes_,
                                 discretize_continuous=True)

    explanation = explainer.explain_instance(sample_input[0], model.predict, num_features=3) #Pass the reshaped sample input to the explain_instance method.
    
    # Get the top 3 contributing features
    top_features = explanation.as_list()
    print("Top 3 contributing features:")
    top_features_2 = [];
    for feature, weight in top_features:
        pattern = r"[a-zA-Z]+"
        words = re.findall(pattern, feature)
        top_features_2.append(words)
   

    try:
        # Use the model to make a prediction
        prediction = model.predict(scaled_features_2)
        predicted_class_index = np.argmax(prediction)

	# Decode the predicted class index to the plant name
        predicted_plant_name = label_encoder.inverse_transform([predicted_class_index])[0]
        
        # Return the prediction
        return jsonify({'result': f"{predicted_plant_name} :: Top contributing features are {top_features_2}"})
    
    except Exception as e:
        return jsonify({'result': f"Error: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True)

