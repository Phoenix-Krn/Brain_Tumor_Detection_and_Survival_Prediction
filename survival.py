import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import joblib

# Load your CSV
df = pd.read_csv(r"D:\Major_Project_(CSE)\Brain_Tumor_Detection\survival_info.csv")  # replace with actual path
print(df.columns)  # Check available columns

# Simulate tumor volume if missing
if 'Tumor_Volume' not in df.columns:
    df['Tumor_Volume'] = np.random.randint(500, 10000, size=len(df))
wp
# Encode categorical
df['Extent_of_Resection'] = LabelEncoder().fit_transform(df['Extent_of_Resection'])

# Features & Target
X = df[['Age', 'Extent_of_Resection', 'Tumor_Volume']]
y = df['Survival_days']

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save
joblib.dump(model, "D:/Major_Project_(CSE)/Brain_Tumor_Detection/rf_survival_model.pkl")

# Optional: Evaluate
y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
