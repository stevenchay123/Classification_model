import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Load dataset
data = pd.read_csv("heart.csv")

# Tangani nilai 0 pada fitur yang tidak logis (misalnya Cholesterol, RestingBP)
for col in ['RestingBP', 'Cholesterol']:
    median_value = data[col].replace(0, pd.NA).median()
    data[col] = data[col].replace(0, median_value)

# Encode fitur kategorikal
label_encoders = {}
categorical_cols = data.select_dtypes(include='object').columns

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Pisahkan fitur dan target
X = data.drop("HeartDisease", axis=1)
y = data["HeartDisease"]

# Standarisasi fitur numerik
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# KNN - cari k terbaik
param_grid = {'n_neighbors': range(1, 31)}
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid.fit(X_train, y_train)
knn_model = grid.best_estimator_

# GNB
gnb_model = GaussianNB()
gnb_model.fit(X_train, y_train)

# Simpan model dan scaler ke dalam folder heart_disease_app
joblib.dump(knn_model, "heart_disease_app/model_knn.pkl")
joblib.dump(gnb_model, "heart_disease_app/model_gnb.pkl")
joblib.dump(scaler, "heart_disease_app/scaler.pkl")
joblib.dump(label_encoders, "heart_disease_app/label_encoders.pkl")

print("Model dan preprocessing berhasil disimpan ke dalam folder heart_disease_app/")
