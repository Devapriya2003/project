from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import numpy as np
import pickle
import joblib
import json

app = Flask(__name__)

# -----------------------------
# App Config
# -----------------------------
app.config["SECRET_KEY"] = "secret123"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# -----------------------------
# Database Models
# -----------------------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(200), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)

class PredictionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    disease_type = db.Column(db.String(50), nullable=False)
    input_values = db.Column(db.Text, nullable=False)   # store JSON string
    prediction_result = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship("User", backref="predictions")

with app.app_context():
    db.create_all()

# -----------------------------
# Load Models
# -----------------------------
lung_model = pickle.load(open("models/lung_model (3).pkl", "rb"))
lung_scaler = pickle.load(open("models/lung_scaler(3).pkl", "rb"))

heart_model = joblib.load("models/heart_model2.pkl")
heart_scaler = joblib.load("models/heart_scaler2.pkl")

kidney_model = joblib.load("models/kidney_model (1).pkl")
kidney_scaler = joblib.load("models/kidney_scaler (1).pkl")

liver_model = pickle.load(open("models/liver_model.pkl", "rb"))

# -----------------------------
# Helper Functions
# -----------------------------
def is_logged_in():
    return "user_id" in session

def save_prediction(user_id, disease_type, input_values, prediction_result):
    history = PredictionHistory(
        user_id=user_id,
        disease_type=disease_type,
        input_values=json.dumps(input_values),   # convert list/dict to string
        prediction_result=prediction_result
    )
    db.session.add(history)
    db.session.commit()

# -----------------------------
# Authentication Routes
# -----------------------------
@app.route("/")
def home():
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name")
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")

        if not name or not username or not email or not password or not confirm_password:
            flash("Please fill all fields.")
            return redirect(url_for("register"))

        if password != confirm_password:
            flash("Passwords do not match.")
            return redirect(url_for("register"))

        existing_user = User.query.filter(
            (User.username == username) | (User.email == email)
        ).first()

        if existing_user:
            flash("Username or email already exists.")
            return redirect(url_for("register"))

        hashed_password = generate_password_hash(password)

        new_user = User(
            name=name,
            username=username,
            email=email,
            password_hash=hashed_password
        )

        db.session.add(new_user)
        db.session.commit()

        flash("Registration successful. Please login.")
        return redirect(url_for("login"))

    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password_hash, password):
            session["user_id"] = user.id
            session["username"] = user.username
            session["name"] = user.name
            flash("Login successful.")
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid username or password.")
            return redirect(url_for("login"))

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.")
    return redirect(url_for("login"))

# -----------------------------
# Dashboard
# -----------------------------
@app.route("/dashboard")
def dashboard():
    if not is_logged_in():
        return redirect(url_for("login"))
    return render_template("dashboard.html", name=session.get("name"))

# -----------------------------
# Lung Page and Prediction
# -----------------------------
@app.route("/lung")
def lung():
    if not is_logged_in():
        return redirect(url_for("login"))
    return render_template("lung.html")

@app.route("/predict_lung", methods=["POST"])
def predict_lung():
    if not is_logged_in():
        return redirect(url_for("login"))

    try:
        input_data = request.form.to_dict()
        features = [float(x) for x in request.form.values()]
        final_input = np.array(features).reshape(1, -1)

        scaled_input = lung_scaler.transform(final_input)
        prediction = lung_model.predict(scaled_input)

        severity = prediction[0]

        if severity == 0:
            result = "COPD Severity: Mild"
        elif severity == 1:
            result = "COPD Severity: Moderate"
        elif severity == 2:
            result = "COPD Severity: Severe"
        else:
            result = "COPD Severity: Very Severe"

        save_prediction(session["user_id"], "Lung", input_data, result)

        return render_template("lung.html", prediction_text=result)

    except Exception as e:
        return render_template("lung.html", prediction_text=f"Error: {str(e)}")

# -----------------------------
# Heart Page and Prediction
# -----------------------------
@app.route("/heart")
def heart():
    if not is_logged_in():
        return redirect(url_for("login"))
    return render_template("heart.html")

@app.route("/predict_heart", methods=["POST"])
def predict_heart():
    if not is_logged_in():
        return redirect(url_for("login"))

    try:
        input_data = request.form.to_dict()
        features = [float(x) for x in request.form.values()]
        final_input = np.array(features).reshape(1, -1)

        scaled_input = heart_scaler.transform(final_input)
        prediction = heart_model.predict(scaled_input)

        if prediction[0] == 1:
            result = "Heart Disease Detected"
        else:
            result = "No Heart Disease"

        save_prediction(session["user_id"], "Heart", input_data, result)

        return render_template("heart.html", prediction_text=result)

    except Exception as e:
        return render_template("heart.html", prediction_text=f"Error: {str(e)}")

# -----------------------------
# Kidney Page and Prediction
# -----------------------------
@app.route("/kidney")
def kidney():
    if not is_logged_in():
        return redirect(url_for("login"))
    return render_template("kidney.html")

@app.route("/predict_kidney", methods=["POST"])
def predict_kidney():
    if not is_logged_in():
        return redirect(url_for("login"))

    try:
        input_data = request.form.to_dict()
        features = [float(x) for x in request.form.values()]
        final_input = np.array(features).reshape(1, -1)

        scaled_input = kidney_scaler.transform(final_input)
        prediction = kidney_model.predict(scaled_input)

        if prediction[0] == 1:
            result = "Kidney Disease Detected"
        else:
            result = "No Kidney Disease"

        save_prediction(session["user_id"], "Kidney", input_data, result)

        return render_template("kidney.html", prediction_text=result)

    except Exception as e:
        return render_template("kidney.html", prediction_text=f"Error: {str(e)}")

# -----------------------------
# Liver Page and Prediction
# -----------------------------
@app.route("/liver")
def liver():
    if not is_logged_in():
        return redirect(url_for("login"))
    return render_template("liver.html")

@app.route("/predict_liver", methods=["POST"])
def predict_liver():
    if not is_logged_in():
        return redirect(url_for("login"))

    try:
        input_data = request.form.to_dict()
        features = [float(x) for x in request.form.values()]
        final_input = np.array(features).reshape(1, -1)

        prediction = liver_model.predict(final_input)

        if prediction[0] == 1:
            result = "Liver Disease Detected"
        else:
            result = "No Liver Disease"

        save_prediction(session["user_id"], "Liver", input_data, result)

        return render_template("liver.html", prediction_text=result)

    except Exception as e:
        return render_template("liver.html", prediction_text=f"Error: {str(e)}")

# -----------------------------
# Retinal Page
# -----------------------------
@app.route("/retinal")
def retinal():
    if not is_logged_in():
        return redirect(url_for("login"))
    return "<h2>Retinal Disease Prediction Page Coming Soon</h2>"

# -----------------------------
# History Page
# -----------------------------
@app.route("/history")
def history():
    if not is_logged_in():
        return redirect(url_for("login"))

    user_history = PredictionHistory.query.filter_by(
        user_id=session["user_id"]
    ).order_by(PredictionHistory.created_at.desc()).all()

    return render_template("history.html", history=user_history, json=json)

if __name__ == "__main__":
    app.run(debug=True)