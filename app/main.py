from flask import Flask, render_template, request
from sklearn.tree import DecisionTreeClassifier
import pickle

app = Flask(__name__)

with open("DrugPredictionModel.pkl", 'br') as model_file:
    model = pickle.load(model_file)
    model: DecisionTreeClassifier


@app.route('/', methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            age = request.form['age']
            sex = request.form['sex']
            bp = request.form['bp']
            cholesterol = request.form['cholesterol']
            na_to_k = request.form['na_to_k']

            prediction = model.predict(
                [[age, sex, bp, cholesterol, na_to_k]])[0]
            return render_template("predict.html", data={"prediction": prediction})
        except KeyError:
            return render_template("predict.html", data={"error": "Please enter all the attributes"})
        except Exception as e:
            return render_template("predict.html", data={"error": "An error occured"})
    return render_template("predict.html", data={})


if __name__ == '__main__':
    app.run("localhost", 5000, debug=True)
