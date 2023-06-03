import numpy as np
import pandas as pd
from flask import Flask, request,render_template
import pickle

app= Flask(__name__)


@app.route('/')
def home():
    return render_template('HouseRentPridiction_2.html')

# prediction function 
def ValuePredictor(to_predict_list): 
    to_predict = np.array(to_predict_list).reshape(1, 8) 
    loaded_model = pickle.load(open("HouseRent.pkl", "rb")) 
    result = loaded_model.predict(to_predict) 
    return result[0] 
  
@app.route('/result', methods = ['POST']) 
def result(): 
    if request.method == 'POST':
        required_fields = {'City':'city', 'BHK':'bhk', 'Furniture Status':'furstatus','Bathroom':'bathroom','Area Type':'areatype','Point of Contact':'pointofcontact','Flat Size':'flatsize','Building Floor':'buildingfloor'}
        errors = []
        for key,value in required_fields.items():
            if value not in request.form or not request.form[value]:
                errors.append(f"{Key} is required.")
        if errors:
            return render_template('result.html', prediction = errors)
        to_dictionary = request.form.to_dict() 
        to_predict_list = request.form.to_dict() 
        to_predict_list = list(to_predict_list.values()) 
        to_predict_list = list(map(int, to_predict_list)) 
        result = ValuePredictor(to_predict_list)                     
        return render_template("result.html", prediction = round(result,2)) 
if __name__ == "__main__":
    app.run(debug=True)
    app.config['TEMPLATES_AUTO_RELOAD'] = True
