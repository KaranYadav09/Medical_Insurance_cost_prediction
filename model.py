import pandas as pd
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv("insurance.csv")

# Encode categorical values
le = LabelEncoder()
data['sex'] = le.fit_transform(data['sex'])  # 0: Female, 1: Male
data['smoker'] = le.fit_transform(data['smoker'])  # 0: No, 1: Yes
data['region'] = le.fit_transform(data['region'])  # 0: Southwest, 1: Southeast, 2: Northwest, 3: Northeast

# Split data
X = data.drop(columns=['charges'])
y = data['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = xgb.XGBRegressor(objective="reg:squarederror", learning_rate=0.1, max_depth=5, n_estimators=100)
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("model.pkl", "wb"))
print("Model trained and saved as model.pkl!")
