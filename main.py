import os
import pickle
from typing import List, Dict, Any
from flask import Blueprint, render_template, session, redirect, url_for, request, flash
import numpy as np

main = Blueprint("main", __name__)

# ---------------------------
# Supported diseases & forms
# ---------------------------
# Expected model filenames (put these in ./models):
#  - diabetes_model.pkl
#  - heart_model.pkl
#  - kidney_model.pkl
#  - liver_model.pkl
#  - breast_cancer_model.pkl
#  - malaria_model.pkl
#  - pneumonia_model.pkl

def _model_path(disease: str) -> str:
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "models", f"{disease}_model.pkl")

def _form_config(disease: str) -> List[Dict[str, Any]]:
    """Return list of field definitions for the requested disease."""
    d = disease.lower()
    if d == "diabetes":
        return [
            {"name":"Pregnancies","label":"Pregnancies","type":"number"},
            {"name":"Glucose","label":"Glucose","type":"number"},
            {"name":"BloodPressure","label":"Blood Pressure","type":"number"},
            {"name":"SkinThickness","label":"Skin Thickness","type":"number"},
            {"name":"Insulin","label":"Insulin","type":"number"},
            {"name":"BMI","label":"BMI","type":"number","step":"0.1"},
            {"name":"DiabetesPedigreeFunction","label":"Diabetes Pedigree Function","type":"number","step":"0.01"},
            {"name":"Age","label":"Age","type":"number"},
        ]
    if d == "heart":
        return [
            {"name":"Age","label":"Age","type":"number"},
            {"name":"Sex","label":"Sex","type":"select","options":["male","female"]},
            {"name":"ChestPain","label":"Chest Pain Type","type":"select","options":["TA","ATA","NAP","ASY"]},
            {"name":"RestingBP","label":"Resting BP","type":"number"},
            {"name":"Cholesterol","label":"Cholesterol","type":"number"},
            {"name":"FastingBS","label":"Fasting Blood Sugar >120 mg/dl?","type":"select","options":["0","1"]},
            {"name":"RestingECG","label":"Resting ECG","type":"select","options":["Normal","ST","LVH"]},
            {"name":"MaxHR","label":"Max Heart Rate","type":"number"},
            {"name":"ExerciseAngina","label":"Exercise Angina","type":"select","options":["N","Y"]},
            {"name":"Oldpeak","label":"Oldpeak (ST depression)","type":"number","step":"0.1"},
            {"name":"ST_Slope","label":"ST Slope","type":"select","options":["Up","Flat","Down"]},
        ]
    if d == "kidney":
        return [
            {"name":"Age","label":"Age","type":"number"},
            {"name":"BloodPressure","label":"Blood Pressure","type":"number"},
            {"name":"SpecificGravity","label":"Specific Gravity","type":"number","step":"0.01"},
            {"name":"Albumin","label":"Albumin","type":"number","step":"0.1"},
            {"name":"Sugar","label":"Sugar","type":"number","step":"0.1"},
            {"name":"RedBloodCells","label":"Red Blood Cells","type":"select","options":["normal","abnormal"]},
            {"name":"PusCell","label":"Pus Cell","type":"select","options":["normal","abnormal"]},
            {"name":"SerumCreatinine","label":"Serum Creatinine","type":"number","step":"0.1"},
            {"name":"Sodium","label":"Sodium","type":"number","step":"0.1"},
            {"name":"Potassium","label":"Potassium","type":"number","step":"0.1"},
        ]
    if d == "liver":
        return [
            {"name":"Age","label":"Age","type":"number"},
            {"name":"Gender","label":"Gender","type":"select","options":["male","female"]},
            {"name":"Total_Bilirubin","label":"Total Bilirubin","type":"number","step":"0.1"},
            {"name":"Direct_Bilirubin","label":"Direct Bilirubin","type":"number","step":"0.1"},
            {"name":"Alkaline_Phosphotase","label":"Alkaline Phosphotase","type":"number"},
            {"name":"Alamine_Aminotransferase","label":"ALT","type":"number"},
            {"name":"Aspartate_Aminotransferase","label":"AST","type":"number"},
            {"name":"Total_Protiens","label":"Total Proteins","type":"number","step":"0.1"},
            {"name":"Albumin","label":"Albumin","type":"number","step":"0.1"},
            {"name":"Albumin_and_Globulin_Ratio","label":"A/G Ratio","type":"number","step":"0.01"},
        ]
    if d == "breast_cancer":
        # minimal example — adjust to your trained features
        return [
            {"name":"mean_radius","label":"Mean Radius","type":"number"},
            {"name":"mean_texture","label":"Mean Texture","type":"number"},
            {"name":"mean_perimeter","label":"Mean Perimeter","type":"number"},
            {"name":"mean_area","label":"Mean Area","type":"number"},
            {"name":"mean_smoothness","label":"Mean Smoothness","type":"number","step":"0.001"},
        ]
    if d == "malaria":
        return [
            {"name":"Parasite_Count","label":"Parasite Count","type":"number"},
            {"name":"RBC_Count","label":"RBC Count","type":"number"},
            {"name":"WBC_Count","label":"WBC Count","type":"number"},
            {"name":"Platelets","label":"Platelets","type":"number"},
        ]
    if d == "pneumonia":
        # for simple demo, use numeric symptoms/codes (real systems use images)
        return [
            {"name":"Age","label":"Age","type":"number"},
            {"name":"Fever","label":"Fever (°C)","type":"number","step":"0.1"},
            {"name":"CoughDays","label":"Days of cough","type":"number"},
            {"name":"BreathRate","label":"Breathing rate","type":"number"},
        ]
    return []


# ---------------------------
# Preprocessing (map selects -> numeric)
# ---------------------------
def _preprocess(disease: str, form: Dict[str, str]) -> List[float]:
    cfg = _form_config(disease)
    # encoders for categorical selects
    enc = {
        "Sex": {"male": 1.0, "female": 0.0},
        "ChestPain": {"TA":0.0,"ATA":1.0,"NAP":2.0,"ASY":3.0},
        "FastingBS": {"0":0.0,"1":1.0},
        "RestingECG": {"Normal":0.0,"ST":1.0,"LVH":2.0},
        "ExerciseAngina": {"N":0.0,"Y":1.0},
        "ST_Slope": {"Up":2.0,"Flat":1.0,"Down":0.0},
        "Gender": {"male":1.0,"female":0.0},
        "RedBloodCells": {"normal":1.0,"abnormal":0.0},
        "PusCell": {"normal":1.0,"abnormal":0.0},
    }

    vals: List[float] = []
    for f in cfg:
        name = f["name"]
        raw = (form.get(name) or "").strip()
        if f["type"] == "select":
            if name in enc:
                v = enc[name].get(raw)
                if v is None:
                    raise ValueError(f"Invalid option for {name}: {raw}")
                vals.append(float(v))
            else:
                # if a select without encoder, try numeric mapping
                vals.append(float(raw))
        else:
            vals.append(float(raw))
    return vals


# ---------------------------
# Probability, meal plan, helpers
# ---------------------------
def _probability_from_model(model, features: List[float]) -> float:
    X = np.array(features, dtype=float).reshape(1, -1)
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)[0][1]
        return float(p * 100.0)
    if hasattr(model, "decision_function"):
        z = float(model.decision_function(X)[0])
        prob = 1.0 / (1.0 + np.exp(-z))
        return prob * 100.0
    pred = int(model.predict(X)[0])
    return 90.0 if pred == 1 else 10.0

def _meal_plan(disease: str, risk_pct: float) -> Dict[str, Dict[str,str]]:
    # simple tailored meal plan; customize to your needs
    disease = disease.lower()
    strict = risk_pct >= 50
    if disease == "diabetes":
        base = {
            "breakfast": "Oats + boiled egg",
            "lunch": "Grilled chicken + quinoa + salad",
            "dinner": "Steamed vegetables + paneer/tofu"
        }
    elif disease == "heart":
        base = {
            "breakfast": "Oatmeal + fruit",
            "lunch": "Salmon/tofu + brown rice + greens",
            "dinner": "Lentil soup + vegetables"
        }
    elif disease == "kidney":
        base = {
            "breakfast": "Low-potassium smoothie + toast",
            "lunch": "Herbed chicken + white rice + veggies",
            "dinner": "Vegetable stir fry (low salt)"
        }
    elif disease == "liver":
        base = {
            "breakfast": "Oats + banana",
            "lunch": "Steamed fish + veggies",
            "dinner": "Vegetable soup + brown rice"
        }
    elif disease == "breast_cancer":
        base = {
            "breakfast": "Greek yogurt + berries",
            "lunch": "Lentil salad + whole grain",
            "dinner": "Grilled fish + greens"
        }
    elif disease == "malaria":
        base = {
            "breakfast": "Light porridge + fruit",
            "lunch": "Simple dal + rice + veggies",
            "dinner": "Broth soup + toast"
        }
    elif disease == "pneumonia":
        base = {
            "breakfast": "Warm porridge + fruit",
            "lunch": "Chicken soup + vegetables",
            "dinner": "Light stew + whole grain"
        }
    else:
        base = {"breakfast":"Balanced breakfast","lunch":"Balanced lunch","dinner":"Balanced dinner"}

    plan = {}
    for i in range(1,8):
        plan[f"Day {i}"] = {
            "breakfast": base["breakfast"],
            "lunch": base["lunch"],
            "dinner": base["dinner"]
        }
    return plan


# ---------------------------
# Routes
# ---------------------------

@main.route("/")
def home():
    return render_template("base.html")

@main.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("auth.login"))
    # show the list of supported diseases
    diseases = ["diabetes","heart","kidney","liver","breast_cancer","malaria","pneumonia"]
    return render_template("dashboard.html", user=session.get("user"), diseases=diseases)


@main.route("/predict/<disease>", methods=["GET","POST"])
def predict(disease):
    if "user" not in session:
        return redirect(url_for("auth.login"))

    disease = disease.lower()
    fields = _form_config(disease)
    if not fields:
        flash("Unknown disease or form not configured.", "error")
        return redirect(url_for("main.dashboard"))

    model_path = _model_path(disease)

    if request.method == "GET":
        return render_template("predict.html", disease=disease.title(), fields=fields)

    # POST: run prediction
    try:
        features = _preprocess(disease, request.form)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        with open(model_path,"rb") as fh:
            model = pickle.load(fh)
        prob = round(_probability_from_model(model, features), 1)
        plan = _meal_plan(disease, prob)
        return render_template("result.html", disease=disease.title(), probability=prob, meal_plan=plan)
    except Exception as e:
        flash(f"Error: {e}", "error")
        return render_template("predict.html", disease=disease.title(), fields=fields)
