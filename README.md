# Chrono-Lock Anti-Spoofing API 🛡️

Chrono-Lock is an AI-powered Anti-Spoofing system designed to detect "Mock Location" (GPS spoofing) fraud. It validates real-world movement by cross-referencing reported GPS speeds with internal hardware sensor data (Accelerometer and Gyroscope variance).

## How it Works
When a user physically moves (e.g., walking or driving), the device naturally experiences vibrations and rotations, resulting in a **high variance** in accelerometer and gyroscope readings. 

If a user attempts to spoof their location to simulate movement (e.g., driving at 15 m/s) while their device is sitting flat on a desk, the hardware sensors will report near-zero variance. The Chrono-Lock AI model uses a **One-Class SVM** anomaly detection algorithm—trained on legitimate movement profiles—to instantly flag this discrepancy as fraud.

## Features
- **Synthetic Data Generation**: Simulates both legitimate and spoofed movement profiles for training.
- **Semi-Supervised Anomaly Detection**: Uses Scikit-Learn's `OneClassSVM` and `StandardScaler` to accurately detect fraudulent sensor combinations.
- **REST API Integration**: Built with **FastAPI** to easily serve predictions to external applications.

## API Endpoint: `/predict`
**Method:** `POST`

**Request Body:**
```json
{
  "gps_speed": 15.0,
  "accel_variance": 0.02,
  "gyro_variance": 0.01
}
```

**Response (Fraud Detected):**
```json
{
  "status": "SPOOFING DETECTED",
  "message": "Anomaly detected: GPS speed does not match physical hardware sensors.",
  "data": {
    "gps_speed": 15.0,
    "accel_variance": 0.02,
    "gyro_variance": 0.01
  }
}
```

## Running Locally

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start the FastAPI server:
```bash
uvicorn api:app --reload
```

3. Visit `http://127.0.0.1:8000/docs` to interact with the API via the built-in Swagger UI.
