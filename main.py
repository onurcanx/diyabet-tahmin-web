from fastapi import FastAPI, HTTPException, Request, Form
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI()

# CORS ayarları
origins = [
    "http://localhost",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")

class DiabetesData(BaseModel):
    pregnancies: float
    glucose: float
    blood_pressure: float
    skin_thickness: float
    insulin: float
    bmi: float
    diabetes_pedigree: float
    age: float

# Eğitilmiş SVM modeli
svm_model = None
# Veri ön işleme scaler
scaler = None
# Veri kümesi
X = None

def train_model():
    global svm_model, scaler, X
    
    # Veri kümesini yükle
    try:
        veri_kumesi = pd.read_csv("C:/Users/onur_/Downloads/diabetes.csv", delimiter=";", decimal=",")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="CSV dosyası bulunamadı")
    
    # Sütunların mevcut olup olmadığını kontrol et
    required_columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
    for col in required_columns:
        if col not in veri_kumesi.columns:
            raise KeyError(f"Sütun '{col}' CSV dosyasında bulunamadı.")
    
    # "Outcome" sütununu düşürerek X ve y'yi ayır
    X = veri_kumesi.drop("Outcome", axis=1)
    y = veri_kumesi["Outcome"]

    # Verileri eğitim ve test setlerine böle
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Verileri standartlaştır
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    tahminn = lr.predict(X_test)
    #print(tahminn)
    # SVM modelini eğit
    svm_model = SVC(kernel='linear', random_state=0)
    svm_model.fit(X_train_scaled, y_train)

    # Doğruluk oranını hesapla ve ekrana yazdır
    train_accuracy = svm_model.score(X_train_scaled, y_train)
    test_accuracy = svm_model.score(X_test_scaled, y_test)
    print("Test Seti Doğruluk Oranı:", test_accuracy)

    predictions = svm_model.predict(X_test_scaled)

@app.on_event("startup")
async def startup_event():
    train_model()

@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict_diabetes(request: Request,
                            pregnancies: float = Form(...),
                            glucose: float = Form(...),
                            blood_pressure: float = Form(...),
                            skin_thickness: float = Form(...),
                            insulin: float = Form(...),
                            bmi: float = Form(...),
                            diabetes_pedigree: float = Form(...),
                            age: float = Form(...)):
    global svm_model, scaler, X
    
    # Girdileri bir veri çerçevesine dönüştür
    user_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]], columns=X.columns)
    
    # Verileri standartlaştır
    user_data_scaled = scaler.transform(user_data)
    
    # Model tarafından tahmin yap
    prediction = svm_model.predict(user_data_scaled)
    
    # Tahmini döndür
    result = "Diyabet riski var" if prediction[0] == 1 else "Diyabet riski yok"
    return templates.TemplateResponse("index.html", {"request": request, "result": result})

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
