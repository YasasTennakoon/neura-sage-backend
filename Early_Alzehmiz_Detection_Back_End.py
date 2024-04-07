from flask import Flask, request, jsonify
from DataFusion import dataFusion 
from DataFusion import imageAnalysis
from werkzeug.utils import secure_filename
import os
import json
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/disease/diagnose', methods=['POST'])
def diagnose_patient():

    data = request.form
    patient_details = data.get('patient')
    image = request.files['image']

    # Saving the image to get the respective path
    filename = secure_filename(image.filename)
    save_path = os.path.join('UplodedImages', filename)
    image.save(save_path)

    print(patient_details)

    #Converting the provide json patient details for a python dictionary
    try:
        data = json.loads(patient_details)
    except json.JSONDecodeError:
        return jsonify({"error": "The entered values are not in the correct format"}), 400
    
    # Printing data for debugging purposes.
    print(data)

    diagnosis_result = dataFusion(data, save_path)
    return jsonify({'diagnosis': diagnosis_result})

@app.route('/disease/image/diagnosis', methods=['POST'])
def diagnose_patient_image():

    data = request.form
    image = request.files['image']

    # Saving the image to get the respective path
    filename = secure_filename(image.filename)
    save_path = os.path.join('UplodedImages', filename)
    image.save(save_path)

    diagnosis_result = imageAnalysis(save_path)
    return jsonify({'diagnosis': diagnosis_result})   


if __name__ == '__main__':
    app.run(debug=True)