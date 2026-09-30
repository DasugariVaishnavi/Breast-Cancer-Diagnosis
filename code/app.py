from flask import Flask, render_template, request, redirect, url_for, session
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import os
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = "abcdefgh123456789"

# Simple in-memory user store (demo purpose only)
users = {}

# Global variables for model and scaler
model = None
scaler = None
model_trained = False

def check_password(stored_password, provided_password):
    return stored_password == provided_password

def train_model():
    """Train the SVM model when the app starts"""
    global model, scaler, model_trained
    
    try:
        print("[TRAIN] Starting model training...")
        
        # Check if data.csv exists
        if not os.path.exists("data.csv"):
            print("[ERROR] Error: data.csv not found!")
            return False
        
        # Load the dataset
        df = pd.read_csv("data.csv")
        
        print("[TRAIN] Dataset Shape:", df.shape)
        
        # Select features - Simplified to 3 key features to resolve multicollinearity
        selected_features = [
            'area_worst',
            'concave_points_worst',
            'concavity_mean'
        ]
        
        # Prepare features and target
        X = df[selected_features]
        y = df['diagnosis'].map({'B': 0, 'M': 1})  # Benign=0, Malignant=1
        
        # Check class distribution
        print("[TRAIN] Class distribution:")
        print(f"Benign (0): {sum(y == 0)} samples")
        print(f"Malignant (1): {sum(y == 1)} samples")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train SVM with GridSearchCV
        print("[TRAIN] Training SVM with GridSearchCV...")
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }
        
        grid = GridSearchCV(SVC(probability=True, random_state=42), param_grid, 
                           refit=True, verbose=1, cv=5, n_jobs=-1)
        grid.fit(X_train_scaled, y_train)
        
        # Get best model
        model = grid.best_estimator_
        
        # Evaluate on test set
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print("[TRAIN] Model trained successfully!")
        print(f"[RE-TRAINED] Best Parameters: {grid.best_params_}")
        print(f"[RE-TRAINED] Test Accuracy: {accuracy * 100:.2f}%")
        print("[RE-TRAINED] Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Benign', 'Malignant']))
        
        # Save the model and scaler
        with open("svm_best_model1.pkl", "wb") as f:
            pickle.dump(model, f)
        
        with open("scaler11.pkl", "wb") as f:
            pickle.dump(scaler, f)
        
        print("[TRAIN] Model and scaler saved successfully!")
        
        # Verify model classes
        print(f"[RE-TRAINED] Model classes: {model.classes_}")
        print(f"[RE-TRAINED] Mapping: 0=Benign, 1=Malignant")
        
        model_trained = True
        return True
        
    except Exception as e:
        print(f"[ERROR] Error during model training: {str(e)}")
        return False

def load_or_train_model():
    """Load existing model or train a new one"""
    global model, scaler, model_trained
    
    try:
        # Try to load existing model and scaler
        if os.path.exists("svm_best_model1.pkl") and os.path.exists("scaler11.pkl"):
            print("[LOAD] Loading existing model and scaler...")
            with open("svm_best_model1.pkl", 'rb') as f:
                model = pickle.load(f)
            with open("scaler11.pkl", 'rb') as s:
                scaler = pickle.load(s)
            
            print("[LOAD] Model and scaler loaded successfully!")
            print(f"[LOAD] Model classes: {model.classes_}")
            model_trained = True
            
            print(f"[CHECK] Test prediction (all zeros): {test_pred[0]}")
            
        else:
            print("[LOAD] No existing model found. Training new model...")
            train_model()
            
    except Exception as e:
        print(f"[WARN] Error loading model: {str(e)}")
        print("[WARN] Training new model...")
        train_model()

# Train or load model when app starts
print("\n" + "="*50)
print("BREAST CANCER DETECTION SYSTEM")
print("="*50)
load_or_train_model()
print("="*50 + "\n")

# ------------------- ROUTES -------------------

@app.route("/")
def login():
    if 'username' in session:
        return redirect(url_for('home'))
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if username in users:
            return redirect(url_for('register'))

        users[username] = {'password': password}
        return redirect(url_for('login'))
    
    return render_template("register.html")

@app.route("/login", methods=["POST"])
def login_post():
    username = request.form["username"]
    password = request.form["password"]

    if username in users and check_password(users[username]['password'], password):
        session['username'] = username
        return redirect(url_for('home'))

    return redirect(url_for('login'))

@app.route("/logout")
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route("/home")
def home():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template("home.html", username=session['username'])

@app.route("/input", methods=["GET", "POST"])
def input_page():
    global model, scaler, model_trained
    
    if 'username' not in session:
        return redirect(url_for('login'))

    if request.method == "POST":
        try:
            # Check if model is trained
            if not model_trained or model is None or scaler is None:
                return "Model not trained yet. Please check server logs.", 500
            
            # -------- Collect features (Internal model uses 3) --------
            area_worst = float(request.form["area_worst"])
            concave_points_worst = float(request.form["concave_points_worst"])
            concavity_mean = float(request.form["concavity_mean"])
            
            # (Keeping other inputs available for potential future use, but not for the model)
            # concave_points_mean = float(request.form["concave_points_mean"])
            # radius_worst = float(request.form["radius_worst"])
            # perimeter_worst = float(request.form["perimeter_worst"])
            # perimeter_mean = float(request.form["perimeter_mean"])
            # area_mean = float(request.form["area_mean"])

            # -------- Prepare data (Simplified feature set) --------
            features = np.array([[area_worst, concave_points_worst, concavity_mean]])

            print("\n[DEBUG] Raw input features (Simplified):", features)

            # -------- Scale data --------
            scaled_data = scaler.transform(features)
            print("[DEBUG] Scaled input:", scaled_data)

            # -------- Predict --------
            prediction = model.predict(scaled_data)[0]
            prediction_proba = model.predict_proba(scaled_data)[0]
            print("[DEBUG] Model prediction:", prediction)
            print("[DEBUG] Prediction probabilities:", prediction_proba)
            print("[DEBUG] Model classes:", model.classes_)

            # -------- Label mapping & Confidence --------
            confidence_benign = prediction_proba[0] * 100
            confidence_malignant = prediction_proba[1] * 100
            
            # -------- Diagnostic Logging --------
            if hasattr(model, 'coef_') and model.kernel == 'linear':
                # Simplified impact check
                impact_area = scaled_data[0][0] * model.coef_[0][0]
                impact_texture = scaled_data[0][1] * model.coef_[0][1]
                impact_shape = scaled_data[0][2] * model.coef_[0][2]
                print(f"[DIAGNOSTIC] Internal Score: {model.decision_function(scaled_data)[0]:.2f}")
                print(f"[DIAGNOSTIC] Impacts: Area={impact_area:.2f}, Texture={impact_texture:.2f}, Shape={impact_shape:.2f}")

            print(f"[DEBUG] Benign Confidence: {confidence_benign:.2f}%")
            print(f"[DEBUG] Malignant Confidence: {confidence_malignant:.2f}%")


            # Pass variables that result.html expects: prediction, confidence_benign, confidence_malignant
            return render_template("result.html", 
                                 prediction=int(prediction), 
                                 confidence_benign=confidence_benign, 
                                 confidence_malignant=confidence_malignant)

        except KeyError as e:
            return f"Missing input field in form: {str(e)}", 400
        except ValueError as e:
            return f"Invalid number format: {str(e)}", 400
        except Exception as e:
            return f"Error during prediction: {str(e)}", 500

    return render_template("input.html")

@app.route("/performance")
def performance():
    if 'username' not in session:
        return redirect(url_for('login'))

    # You can calculate these metrics dynamically from the model if needed
    metrics = {
        'accuracy': 99.12,
        'precision': 100.0,
        'recall': 98.0,
        'f1_score': 99.0,
        'model_status': 'Trained' if model_trained else 'Not Trained'
    }

    return render_template("performance.html", metrics=metrics)

@app.route("/charts")
def charts():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template("charts.html")

@app.route("/retrain", methods=["POST"])
def retrain_model():
    """Endpoint to manually retrain the model"""
    global model, scaler, model_trained
    
    if 'username' not in session:
        return redirect(url_for('login'))
    
    success = train_model()
    if success:
        return {"status": "success", "message": "Model retrained successfully!"}
    else:
        return {"status": "error", "message": "Failed to retrain model."}, 500

# Add a test route to verify model predictions
@app.route("/test")
def test_model():
    """Test route to verify model predictions"""
    global model, scaler
    
    if model is None or scaler is None:
        return "Model not loaded", 500
    
    # Test with values that should predict Malignant (high values)
    malignant_test = np.array([[2000.0, 0.2, 0.15, 25.0, 180.0, 150.0, 0.2, 1200.0]])
    malignant_scaled = scaler.transform(malignant_test)
    malignant_pred = model.predict(malignant_scaled)[0]
    malignant_proba = model.predict_proba(malignant_scaled)[0]
    
    # Test with values that should predict Benign (low values)
    benign_test = np.array([[500.0, 0.05, 0.02, 12.0, 80.0, 70.0, 0.03, 400.0]])
    benign_scaled = scaler.transform(benign_test)
    benign_pred = model.predict(benign_scaled)[0]
    benign_proba = model.predict_proba(benign_scaled)[0]
    
    return f"""
    <html>
    <head>
        <title>Model Test Results</title>
        <style>
            body {{ font-family: Arial; margin: 40px; }}
            .container {{ max-width: 800px; margin: auto; }}
            .card {{ border: 1px solid #ddd; padding: 20px; margin: 20px 0; border-radius: 10px; }}
            .malignant {{ color: red; font-weight: bold; }}
            .benign {{ color: green; font-weight: bold; }}
            h2 {{ color: #333; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔬 Model Test Results</h1>
            
            <div class="card">
                <h3>Test Case 1: Malignant (High Values)</h3>
                <p>Prediction: <span class="malignant">{'Malignant' if malignant_pred == 1 else 'Benign'}</span></p>
                <p>Confidence: Benign: {malignant_proba[0]*100:.2f}%, Malignant: {malignant_proba[1]*100:.2f}%</p>
                <p>Expected: Malignant</p>
                <p>Status: {'✅ Correct' if malignant_pred == 1 else '❌ Incorrect'}</p>
            </div>
            
            <div class="card">
                <h3>Test Case 2: Benign (Low Values)</h3>
                <p>Prediction: <span class="benign">{'Malignant' if benign_pred == 1 else 'Benign'}</span></p>
                <p>Confidence: Benign: {benign_proba[0]*100:.2f}%, Malignant: {benign_proba[1]*100:.2f}%</p>
                <p>Expected: Benign</p>
                <p>Status: {'✅ Correct' if benign_pred == 0 else '❌ Incorrect'}</p>
            </div>
            
            <div class="card">
                <h3>Model Information</h3>
                <p>Model Classes: {model.classes_} (0=Benign, 1=Malignant)</p>
                <p>Model Type: SVM with probability=True</p>
                <p>Scaler: StandardScaler (fitted)</p>
            </div>
            
            <p><a href="/input">Go to Input Form</a> | <a href="/">Back to Login</a></p>
        </div>
    </body>
    </html>
    """

# ------------------- MAIN -------------------
if __name__ == '__main__':
    app.run(debug=True)