from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os

app = FastAPI()

# السماح بالطلبات من واجهة الويب
origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# تحميل النموذج
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "RiceDiseaseModel.h5")

if os.path.exists(model_path):
    MODEL = tf.keras.models.load_model(model_path)
    CLASS_NAMES = [
        'Bacterial Leaf Blight', 'Brown Spot', 'Healthy Rice Leaf',
        'Leaf Blast', 'Leaf scald', 'Narrow Brown Leaf Spot',
        'Neck_Blast', 'Rice Hispa', 'Sheath Blight'
    ]
else:
    raise FileNotFoundError(f"Model file not found at {model_path}")

# الصفحة الرئيسية
@app.get("/")
async def root():
    return {"message": "Welcome to the Rice Disease Prediction API!"}

# اختبار أن التطبيق يعمل
@app.get("/ping")
async def ping():
    return "Hello, I am alive"

# تعديل `read_file_as_image` لتكون متوافقة مع التدريب
def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data))  # تحميل الصورة
    img_array = tf.keras.preprocessing.image.img_to_array(image)  # تحويلها إلى مصفوفة
    img_array = tf.expand_dims(img_array, 0)  # إضافة بعد إضافي
    return img_array  

# مسار التنبؤ
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())  # تجهيز الصورة
    predictions = MODEL.predict(image)  # إجراء التنبؤ
    predicted_class = CLASS_NAMES[np.argmax(predictions, axis=1)[0]]  # استخراج الفئة المتوقعة
    confidence = float(np.max(predictions, axis=1)[0])  # استخراج نسبة الثقة

    return {
        'class': predicted_class,
        'confidence': confidence
    }

# تشغيل التطبيق
if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8080)
