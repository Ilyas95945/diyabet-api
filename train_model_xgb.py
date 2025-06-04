import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from joblib import dump

# 📥 CSV'den veriyi oku
df = pd.read_csv("Multiclass Diabetes Dataset.csv")

# ✅ Gerekli sütunları filtrele
df = df[["AGE", "HbA1c", "BMI", "Gender", "Chol", "TG", "HDL", "LDL", "Cr", "Urea", "Class"]]

# 🔧 Özellikler ve hedef değişkeni ayır
X = df.drop("Class", axis=1)
y = df["Class"]

# 🔀 Eğitim-test bölmesi (zorunlu değil ama kontrol amaçlı)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🤖 Modeli oluştur
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

# 🎯 Modeli eğit
model.fit(X_train, y_train)

# 🧾 Test sonuçları (terminalde gösterilecek)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# 💾 Modeli kaydet
dump(model, "model.pkl")
print("✅ XGBoost modeli başarıyla eğitildi ve 'model.pkl' olarak kaydedildi.")
