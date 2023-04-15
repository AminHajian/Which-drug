from flask import Flask, render_template, request
from sklearn.tree import DecisionTreeClassifier
import pickle

app = Flask(__name__)

with open("DrugPredictionModel.pkl", 'br') as model_file:
    model = pickle.load(model_file)
    model:DecisionTreeClassifier
    
@app.route('/predict', methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        age = request.form['age']
        sex = request.form['sex']
        bp = request.form['bp']
        cholesterol = request.form['cholesterol']
        na_to_k = request.form['na_to_k']
        
        prediction = model.predict([[age, sex, bp, cholesterol, na_to_k]])
        return render_template("predict.html", data={"prediction": prediction})
    return render_template("predict.html", data={})
    
if __name__ == '__main__':
    app.run("0.0.0.0", 5000, debug=True)