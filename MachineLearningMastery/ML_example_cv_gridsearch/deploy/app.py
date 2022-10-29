import os
os.chdir(r'D:/soft_code/machine_learning/machinelearning/ML_example_cv_gridsearch/deploy/')
from flask import Flask, jsonify, request
from flask_restful import Api, Resource
from model.Train import train_model
import joblib


app = Flask(__name__)
@app.route('/invocation',methods=["POST"])
# api = Api(app)

def invocation():
    if not os.path.isfile('model/iris-model.model'):
        train_model()
    model = joblib.load('model/iris-model.model')
    posted_data = request.get_json()
    sepal_length = posted_data['sepal_length']
    sepal_width = posted_data['sepal_width']
    petal_length = posted_data['petal_length']
    petal_width = posted_data['petal_width']
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]
    if prediction == 0:
        predicted_class = 'Iris-setosa'
    elif prediction == 1:
        predicted_class = 'Iris-versicolor'
    else:
        predicted_class = 'Iris-virginica'

    return jsonify({
        'Prediction': str(predicted_class)
    })

# api.add_resource(MakePrediction, '/predict')

if __name__ == '__main__':
   app.run(host='0.0.0.0', port=8080)