import xgboost as xgb
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("insurance.csv")

# Handle missing values (if any)
df = df.dropna()

# Convert categorical columns to numeric
df["smoker"] = df["smoker"].map({"yes": 1, "no": 0})
df["gender"] = df["sex"].map({"male": 0, "female": 1})  # Corrected gender encoding

# Encode categorical 'region'
label_encoder = LabelEncoder()
df["region"] = label_encoder.fit_transform(df["region"])

# Define features and target
X = df[["age", "bmi", "children", "gender", "smoker", "region"]]
y = df["charges"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling (Improves prediction accuracy)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train XGBoost model with optimized parameters
model = xgb.XGBRegressor(
    n_estimators=250, 
    learning_rate=0.05, 
    max_depth=5, 
    subsample=0.8, 
    colsample_bytree=0.8, 
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print(f"📊 Model Evaluation Results:")
print(f"✅ R² Score: {r2:.4f}")
print(f"✅ MAE: {mae:.2f}")
print(f"✅ RMSE: {rmse:.2f}")

# Save Model and Scaler safely
try:
    with open("xgboost_model.pkl", "wb") as file:
        pickle.dump(model, file)

    with open("scaler.pkl", "wb") as file:
        pickle.dump(scaler, file)

    print("✅ Model and scaler successfully trained and saved!")

except Exception as e:
    print(f"❌ Error saving model or scaler: {e}")
