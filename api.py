from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from chrono_lock_ai import generate_synthetic_data, train_model

app = FastAPI(
    title="Chrono-Lock Anti-Spoofing API",
    description="API to detect Mock Location GPS spoofing based on hardware sensor variance.",
    version="1.0.0"
)

# Global variable to hold our trained model
model = None

# Define the data schema that the API expects
class SensorData(BaseModel):
    gps_speed: float
    accel_variance: float
    gyro_variance: float

@app.on_event("startup")
def load_model():
    """
    On startup, generate data and train the AI model so it's ready for inference.
    """
    global model
    print("Initializing Chrono-Lock AI Model...")
    df = generate_synthetic_data(n_samples=2000)
    model = train_model(df)
    print("Model loaded and ready for predictions.")

@app.post("/predict")
def predict_spoofing(data: SensorData):
    """
    Endpoint to predict if the given sensor data is legitimate or spoofed.
    """
    # Convert input to DataFrame for the model
    input_df = pd.DataFrame([{
        "gps_speed": data.gps_speed,
        "accel_variance": data.accel_variance,
        "gyro_variance": data.gyro_variance
    }])
    
    # Model prediction: 1 = Legitimate, -1 = Anomaly/Spoofed
    prediction = model.predict(input_df)[0]
    
    if prediction == 1:
        status = "VERIFIED"
        message = "Legitimate movement detected."
    else:
        status = "SPOOFING DETECTED"
        message = "Anomaly detected: GPS speed does not match physical hardware sensors."
        
    return {
        "status": status,
        "message": message,
        "data": data.dict()
    }
