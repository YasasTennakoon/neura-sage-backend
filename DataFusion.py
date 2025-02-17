from joblib import load
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import os

# The following Includes the path representations of the Models
# Text classification Model

text_ensemble_model_path = os.path.join(os.path.dirname(__file__), 'Models', 'voting_classifier.joblib')

# The image classification tasks includes with three models,
#   1. DenceNet Model
#   2. EfficeintNet Model
#   3. MetaModel - This model is created using the above two models

image_classification_model1_path = os.path.join(os.path.dirname(__file__), 'Models', 'DenceNet.h5')
image_classification_model2_path = os.path.join(os.path.dirname(__file__), 'Models', 'EfficeintNet.h5')
image_classification_meta_model_path = os.path.join(os.path.dirname(__file__), 'Models', 'MetaModel.h5')

# Loading the Models

voting_classifier_ensemble_model = load(text_ensemble_model_path)
dence_net_model = load_model(image_classification_model1_path)
efficient_net_model = load_model(image_classification_model2_path)
meta_model = load_model(image_classification_meta_model_path)


# Defining a function for idnetifying stage of the patient when the classification is conducted.

def stage(number):
    if number == 0:
        return 'Non Demented'
    elif number == 1:
        return 'Mild AD'
    elif number == 2:
        return 'Moderate AD'
    elif number == 3:
        return 'Very Mild AD'
    else:
        return 'Error in Prediction'
    
#Creating the object for the data fusion process containing the combined probility values alog with the image and textual classification probability values.
class FinalClassificationResult:
    def __init__(self, image_probs, text_probs, combined_probs, final_decision):
        self.image_probs = image_probs
        self.text_probs = text_probs
        self.combined_probs = combined_probs
        self.final_decision = final_decision

    def to_dict(self):
        def format_probs(probs):
            probs_array = np.array(probs)
            rounded_probs = []

            for prob in np.nditer(probs_array, flags=['refs_ok']):
                rounded_prob = float(np.round(prob * 100, 2))
                rounded_probs.append(rounded_prob)

            if len(probs_array.shape) > 1:
                rounded_probs = np.reshape(rounded_probs, probs_array.shape).tolist()
            return rounded_probs

        return {
            "image_probs": format_probs(self.image_probs),
            "text_probs": format_probs(self.text_probs),
            "combined_probs": format_probs(self.combined_probs),
            "final_decision": self.final_decision
        }

#Creating the object for the data fusion process containing the image classification probability values.
class FinalImageClassifcationResult:
     def __init__(self, image_probs, final_decision):
        self.image_probs = image_probs
        self.final_decision = final_decision

     def to_dict_image(self):
        def format_image_probs(probs):
            probs_array = np.array(probs)
            rounded_probs = []

            for prob in np.nditer(probs_array, flags=['refs_ok']):
                rounded_prob = float(np.round(prob * 100, 2))
                rounded_probs.append(rounded_prob)

            if len(probs_array.shape) > 1:
                rounded_probs = np.reshape(rounded_probs, probs_array.shape).tolist()
            return rounded_probs

        return {
                "image_probs": format_image_probs(self.image_probs),
                "final_decision": self.final_decision
            }   

# Function for the data fusion process  
def dataFusion(new_patient_data, image_path):

    # 2. Image data
    img = Image.open(image_path)

    #Image data preprocessing and model predictions 
        
    x = np.array(img.resize((128,128)))
    x = x.reshape(1,128,128,3)

    #Test data preprocessing
    new_patient_df = pd.DataFrame(new_patient_data)

    # Predicting from each class and combining the prediction to create the input for the metamodel
    pred1 = dence_net_model.predict(x)
    pred2 = efficient_net_model.predict(x)
    preds_combined = np.concatenate([pred1, pred2], axis=-1)

    # Retriving the probabilistics values for each model
    image_probs = meta_model.predict_on_batch(preds_combined)  
    text_probs = voting_classifier_ensemble_model.predict_proba(new_patient_df)

    # Regraging the textual probabilities.
    original_order = ['Mild AD', 'Moderate AD', 'Not Determined', 'Very Mild AD']
    desired_order = ['Not Determined', 'Mild AD', 'Moderate AD', 'Very Mild AD']

    rearranged_probabilities = text_probs[:, [original_order.index(class_) for class_ in desired_order]]

    print("Image probabilities:", image_probs)
    print("Text probabilities:", rearranged_probabilities)

    combined_probs = np.array([(i_prob + t_prob) / 2 for i_prob, t_prob in zip(image_probs, rearranged_probabilities)])


    print("Combined probabilities:", combined_probs)

    stage_index = np.argmax(combined_probs)
    final_decision = stage(stage_index)

    result = FinalClassificationResult(image_probs, rearranged_probabilities, combined_probs, final_decision)
    return result.to_dict()

#Finction for the only image classifiation proceess.
def imageAnalysis(image_path):
    # 2. Image data
    img = Image.open(image_path)

    #Image data preprocessing and model predictions 
        
    x = np.array(img.resize((128,128)))
    x = x.reshape(1,128,128,3)


    # Predicting from each class and combining the prediction to create the input for the metamodel
    pred1 = dence_net_model.predict(x)
    pred2 = efficient_net_model.predict(x)
    preds_combined = np.concatenate([pred1, pred2], axis=-1)

    # Retriving the probabilistics values for image model
    image_probs = meta_model.predict_on_batch(preds_combined) 

    stage_index = np.argmax(image_probs)
    final_decision = stage(stage_index)

    image_result = FinalImageClassifcationResult(image_probs,final_decision)
    return image_result.to_dict_image()