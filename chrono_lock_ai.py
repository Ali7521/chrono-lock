import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import logging

# Configure logging to output to terminal clearly
logging.basicConfig(level=logging.INFO, format='%(message)s')

def generate_synthetic_data(n_samples=2000):
    """
    Generates a synthetic dataset containing both legitimate movement and spoofed GPS movement.
    
    Feature Engineering Context:
    ----------------------------
    When a person is physically moving (walking, driving), their phone experiences physical vibrations.
    This causes high variance in Accelerometer (linear acceleration) and Gyroscope (rotation) readings.
    
    If someone uses a Mock Location app to spoof their GPS, the system might see the GPS speed as 
    15 m/s (simulating driving), but if the phone is sitting flat on a desk, the Accelerometer 
    and Gyroscope variances will be near zero. This mismatch is the key to detecting spoofing!
    """
    logging.info("--- [1] GENERATING SYNTHETIC DATA ---")
    np.random.seed(42)
    
    # Let's say 80% of our data is legitimate, 20% is mock location (spoofed)
    n_legit = int(n_samples * 0.8)
    n_spoofed = n_samples - n_legit
    
    # ------------------
    # 1. Legitimate Data
    # ------------------
    # Speed > 0, Accelerometer Variance = High, Gyroscope Variance = High
    legit_speed = np.random.uniform(1, 30, n_legit) # m/s
    legit_accel_var = np.random.uniform(0.5, 5.0, n_legit)
    legit_gyro_var = np.random.uniform(0.1, 2.0, n_legit)
    
    # ------------------
    # 2. Spoofed Data (Fraud)
    # ------------------
    # Speed > 0, Accelerometer Variance = Near Zero, Gyroscope Variance = Near Zero
    spoofed_speed = np.random.uniform(1, 30, n_spoofed) # m/s
    spoofed_accel_var = np.random.uniform(0.0, 0.05, n_spoofed) # Phone is on a table
    spoofed_gyro_var = np.random.uniform(0.0, 0.02, n_spoofed)
    
    # Combine datasets
    speed = np.concatenate([legit_speed, spoofed_speed])
    accel_var = np.concatenate([legit_accel_var, spoofed_accel_var])
    gyro_var = np.concatenate([legit_gyro_var, spoofed_gyro_var])
    
    # Labels (1 for inlier/legit, -1 for outlier/spoofed) - IsolationForest uses these labels
    labels = np.concatenate([np.ones(n_legit), -1 * np.ones(n_spoofed)]) 
    
    df = pd.DataFrame({
        'gps_speed': speed,
        'accel_variance': accel_var,
        'gyro_variance': gyro_var,
        'is_spoofed': labels == -1 # Boolean for readability
    })
    
    # Shuffle the dataset
    df = df.sample(frac=1).reset_index(drop=True)
    
    logging.info(f"Generated {n_samples} samples.")
    logging.info(f"Legitimate records: {n_legit} | Spoofed records: {n_spoofed}\n")
    return df

def train_model(df):
    """
    Trains a One-Class SVM anomaly detection model.
    We train it ONLY on legitimate data so it learns the "normal" profile.
    Anything that deviates from this (like zero-variance mock location) is flagged.
    """
    from sklearn.svm import OneClassSVM
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    
    logging.info("--- [2] TRAINING AI MODEL (ONE-CLASS SVM) ---")
    features = ['gps_speed', 'accel_variance', 'gyro_variance']
    
    # Train only on the legitimate data (semi-supervised anomaly detection)
    X_train = df[df['is_spoofed'] == False][features]
    
    # We use a pipeline to scale features so speed (m/s) doesn't dominate variance
    model = make_pipeline(
        StandardScaler(),
        OneClassSVM(nu=0.05, kernel="rbf", gamma="scale")
    )
    model.fit(X_train)
    
    logging.info("Model successfully trained on legitimate sensor data.\n")
    return model

def test_inference(model, test_cases):
    """
    Tests the model with specific scenarios passed as dictionaries.
    """
    logging.info("--- [3] RUNNING INFERENCE TESTS ---")
    features = ['gps_speed', 'accel_variance', 'gyro_variance']
    
    df_test = pd.DataFrame(test_cases)
    X_test = df_test[features]
    
    # Predict (-1 = anomaly/spoof, 1 = normal/legit)
    predictions = model.predict(X_test)
    
    for i, pred in enumerate(predictions):
        status = "✅ VERIFIED" if pred == 1 else "🚨 SPOOFING DETECTED"
        scenario = test_cases[i]['scenario']
        gps_speed = test_cases[i]['gps_speed']
        accel = test_cases[i]['accel_variance']
        gyro = test_cases[i]['gyro_variance']
        
        logging.info(f"Scenario: {scenario}")
        logging.info(f"  -> Data: [Speed: {gps_speed} m/s | Accel Var: {accel} | Gyro Var: {gyro}]")
        logging.info(f"  -> Result: {status}\n")

if __name__ == "__main__":
    logging.info("🚀 INITIALIZING CHRONO-LOCK AI (PHASE 1)\n")
    
    # 1. Generate Synthetic Data
    df_simulated = generate_synthetic_data(n_samples=2000)
    
    # 2. Train the Model
    model = train_model(df_simulated)
    
    # 3. Test with custom inference scenarios
    test_scenarios = [
        {
            'scenario': 'User walking down the street',
            'gps_speed': 1.5,
            'accel_variance': 3.5,
            'gyro_variance': 1.1
        },
        {
            'scenario': 'User driving on a highway',
            'gps_speed': 25.0,
            'accel_variance': 2.0,
            'gyro_variance': 0.8
        },
        {
            'scenario': 'Mock Location App (Phone flat on desk, simulating running)',
            'gps_speed': 5.0,
            'accel_variance': 0.01,
            'gyro_variance': 0.005
        },
        {
            'scenario': 'Teleporting (High speed GPS jump, no physical movement)',
            'gps_speed': 120.0,
            'accel_variance': 0.03,
            'gyro_variance': 0.01
        }
    ]
    
    test_inference(model, test_scenarios)
    logging.info("Phase 1 Complete. Awaiting approval to move to Phase 2.")
