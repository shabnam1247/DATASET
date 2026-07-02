from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict")
def predict():
    return render_template("predict.html")


@app.route("/result", methods=["POST"])
def result():
    try:
        area = float(request.form["area"])
        rooms = int(request.form["rooms"])
        year = int(request.form["year"])

        location = request.form["location"]
        street = request.form["street"]
        furnishing = request.form["furnishing"]
        property = request.form["property"]
        pool = request.form["pool"]

        location = encoders["Location"].transform([location])[0]
        street = encoders["Street_Type"].transform([street])[0]
        furnishing = encoders["Furnishing"].transform([furnishing])[0]
        property = encoders["Property_Type"].transform([property])[0]
        pool = encoders["Has_Pool"].transform([pool])[0]

        data = np.array([[area,
                          rooms,
                          year,
                          location,
                          street,
                          furnishing,
                          property,
                          pool]])

        prediction = model.predict(data)[0]

        return render_template(
            "result.html",
            prediction=f"{prediction:,.2f}"
        )

    except Exception as e:
        return f"<h2>Error</h2><p>{e}</p>"


if __name__ == "__main__":
    app.run(debug=True)