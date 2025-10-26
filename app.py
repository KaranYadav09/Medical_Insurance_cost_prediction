import os
import numpy as np
import pickle
import requests
from flask import Flask, render_template, request, redirect, url_for, session, send_file
from fpdf import FPDF
from datetime import datetime
from config import SHEETDB_URL_1, SHEETDB_URL_2, SECRET_KEY
from flask import send_from_directory

app = Flask(__name__, template_folder="templates")
app.secret_key = SECRET_KEY  # Security key for session management

# Load trained model and scaler
with open("xgboost_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Define region mapping
region_mapping = {
    "southwest": 1,
    "southeast": 2,
    "northwest": 3,
    "northeast": 4
}

# --------------------- AUTHENTICATION FUNCTION ---------------------
def authenticate_user(email, password):
    try:
        response = requests.get(f"{SHEETDB_URL_1}/search?email={email}")
        
        if response.status_code == 200:
            data = response.json()
            if data and "password" in data[0] and data[0]["password"] == password:
                return True  # Valid credentials
    except Exception as e:
        print(f"Error in authentication: {e}")

    return False  # Invalid credentials

# --------------------- SIGNUP ---------------------
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        password = request.form["password"]

        # Save user data to SheetDB
        user_data = {"name": name, "email": email, "password": password}
        response = requests.post(SHEETDB_URL_1, json={"data": [user_data]})

        if response.status_code == 201:
            return redirect(url_for("signin"))
        return "Signup failed!"

    return render_template("signup.html")

# --------------------- LOGIN ---------------------
@app.route("/signin", methods=["GET", "POST"])
def signin():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        if authenticate_user(email, password):
            session["user"] = email
            return redirect(url_for("home"))
        return "Invalid credentials!"  # Updated error message

    return render_template("signin.html")

# --------------------- LOGOUT ---------------------
@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("signin"))

# --------------------- HOME ---------------------
@app.route("/")
def home():
    if "user" not in session:
        return redirect(url_for("signin"))
    return render_template("home.html")

# --------------------- PRICING ---------------------
@app.route('/pricing')
def pricing():
    return render_template('pricing.html')

# --------------------- SUPPORT ---------------------
@app.route('/support')
def support():
    return render_template('support.html')

# --------------------- PREDICTION ---------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "user" not in session:
            return redirect(url_for("signin"))

        # Get user input
        age = int(request.form.get("age", 0))
        if age < 18:
            return render_template("predict.html", error="Prediction is only available for users aged 18 or above.")
        bmi = float(request.form.get("bmi", 0.0))
        children = int(request.form.get("children", 0))
        gender = request.form.get("gender", "0")
        smoker = request.form.get("smoker", "no")
        region = request.form.get("region", "southwest")

        # Encoding categorical variables
        gender_encoded = 1 if gender == "1" else 0
        smoker_encoded = 1 if smoker == "yes" else 0
        region_encoded = region_mapping.get(region, 1)

        # Prepare input for prediction
        input_features = np.array([[age, bmi, children, gender_encoded, smoker_encoded, region_encoded]])
        input_features_scaled = scaler.transform(input_features)

        # Make prediction
        prediction = round(float(model.predict(input_features_scaled)[0]), 2)
        prediction_inr = round(prediction * 83, 2)

        # Save prediction details to SHEETDB_URL_2
        user_email = session["user"]
        prediction_data = {
            "email": user_email,
            "age": age,
            "bmi": bmi,
            "children": children,
            "gender": "Female" if gender == "1" else "Male",
            "smoker": smoker,
            "region": region,
            "prediction_usd": prediction,
            "prediction_inr": prediction_inr,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        requests.post(SHEETDB_URL_2, json={"data": [prediction_data]})

        return render_template("predict.html",
                               prediction_usd=prediction,
                               prediction_inr=prediction_inr,
                               age=age,
                               bmi=bmi,
                               children=children,
                               gender=gender,
                               smoker=smoker,
                               region=region)

    except Exception as e:
        return f"Error in Prediction: {e}"

# --------------------- PDF GENERATION ---------------------
def generate_pdf(age, bmi, children, gender, smoker, region, prediction_usd, prediction_inr):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title Styling
    pdf.set_font("Arial", style="B", size=22)
    pdf.set_text_color(25, 25, 112)  # Dark Blue
    pdf.cell(200, 10, "Medical Cost Prediction Report", ln=True, align="C")
    pdf.ln(10)

    # Date
    pdf.set_font("Arial", size=12)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(200, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
    pdf.ln(15)

    # User Details Table
    pdf.set_font("Arial", style="B", size=14)
    pdf.set_fill_color(200, 200, 200)  # Light Gray
    pdf.cell(0, 10, "User Information", ln=True, align="L", fill=True)
    pdf.ln(5)

    pdf.set_font("Arial", size=12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, f"Age: {age}", ln=True)
    pdf.cell(0, 8, f"BMI: {bmi}", ln=True)
    pdf.cell(0, 8, f"Children: {children}", ln=True)
    pdf.cell(0, 8, f"Gender: {'Male' if gender == '0' else 'Female'}", ln=True)
    pdf.cell(0, 8, f"Smoker: {smoker.capitalize()}", ln=True)
    pdf.cell(0, 8, f"Region: {region.capitalize()}", ln=True)
    pdf.ln(10)

    # Cost Prediction
    pdf.set_font("Arial", style="B", size=14)
    pdf.set_fill_color(200, 200, 200)
    pdf.cell(0, 10, "Predicted Cost", ln=True, align="L", fill=True)
    pdf.ln(5)

    pdf.set_font("Arial", size=12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, f"Estimated Cost (USD): ${prediction_usd}", ln=True)
    pdf.cell(0, 8, f"Estimated Cost (INR): INR {prediction_inr}", ln=True)  # FIX: Replaced ₹ with "INR"
    pdf.ln(10)

    # Health Suggestions
    pdf.set_font("Arial", style="B", size=14)
    pdf.set_fill_color(200, 200, 200)
    pdf.cell(0, 10, "Health Suggestions", ln=True, align="L", fill=True)
    pdf.ln(5)

    pdf.set_font("Arial", size=12)
    pdf.set_text_color(0, 0, 0)

    # Suggestion Based on BMI
    if bmi < 18.5:
        pdf.cell(0, 8, "You are underweight. Consider a balanced diet with healthy fats.", ln=True)
    elif 18.5 <= bmi <= 24.9:
        pdf.cell(0, 8, "Your BMI is normal. Maintain a balanced diet and regular exercise.", ln=True)
    elif 25 <= bmi <= 29.9:
        pdf.cell(0, 8, "You are overweight. Consider reducing sugar intake and increasing exercise.", ln=True)
    else:
        pdf.cell(0, 8, "You are obese. Consult a doctor and adopt a healthier lifestyle.", ln=True)

    # Suggestion for Smokers
    if smoker.lower() == "yes":
        pdf.cell(0, 8, "Smoking increases health risks. Consider quitting for a healthier life.", ln=True)

    pdf.ln(10)

    # Save PDF
    pdf_filename = "medical_cost_report.pdf"
    pdf.output(pdf_filename)

    return pdf_filename

@app.route('/download-pdf', methods=['POST'])
def download_pdf():
    try:
        # Retrieve form data
        age = request.form['age']
        gender = request.form['gender']
        bmi = float(request.form['bmi'])  # Convert to float
        children = request.form['children']
        smoker = request.form['smoker']
        region = request.form['region']
        prediction_usd = request.form['prediction_usd']
        prediction_inr = request.form['prediction_inr']

        # Generate PDF
        pdf_path = generate_pdf(age, bmi, children, gender, smoker, region, prediction_usd, prediction_inr)

        return send_file(pdf_path, as_attachment=True)

    except Exception as e:
        return f"Error generating PDF: {str(e)}"

@app.route("/mobile_app")
def mobile_app():
    return render_template("mobile_app.html")

@app.route('/download-app')
def download_app():
    return send_from_directory(directory='static', path='micp.apk', as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
