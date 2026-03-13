import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Load Data
path = r"C:\Users\mavs2\Documents\ML algorithms project\stroke.csv"
df = pd.read_csv(path)

# --- DATA CLEANING ---
# BMI has missing values; filling with median preserves the distribution
df['bmi'] = df['bmi'].fillna(df['bmi'].median())
df = df.drop(columns=['id'])

# --- ENCODING ---
le = LabelEncoder()
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
for col in categorical_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# --- SEPARATE X AND Y ---
X = df.drop(columns=['stroke'])
y = df['stroke']

# --- FEATURE SELECTION ---
# Selecting the top predictors: Age, Glucose, and Hypertension
selected_features = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']
X = X[selected_features]

# --- FEATURE STANDARDISATION ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- TRAIN/TEST SPLIT ---
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- TRAIN BALANCED DECISION TREE ---
# class_weight='balanced' ensures the model focuses on the minority stroke cases
model = DecisionTreeClassifier(
    criterion='entropy', 
    max_depth=3, 
    class_weight='balanced', 
    random_state=42
)
model.fit(X_train, y_train)

# --- EVALUATION ---
y_pred = model.predict(X_test)
print(f"Balanced Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --- VISUALISATION ---
plt.figure(figsize=(18, 10))
plot_tree(model, 
          feature_names=selected_features, 
          class_names=['No Stroke', 'Stroke'], 
          filled=True, 
          rounded=True, 
          fontsize=10)
plt.title("Balanced Decision Tree for Clinical Stroke Prediction")
plt.show()