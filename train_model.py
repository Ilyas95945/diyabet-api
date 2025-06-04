import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

# CSV dosyasını oku
df = pd.read_csv("Multiclass Diabetes Dataset.csv")

# Sadece gerekli sütunları al
df = df[["AGE", "HbA1c", "BMI", "Class"]]

# Özellikler ve etiket
X = df[["AGE", "HbA1c", "BMI"]]
y = df["Class"]

# Modeli eğit
model = RandomForestClassifier()
model.fit(X, y)

# Modeli dosyaya kaydet
dump(model, "model.pkl")
print("✅ Model başarıyla 'Multiclass Diabetes Dataset.csv' üzerinden eğitildi.")
