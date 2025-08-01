import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Step 1: Simulate Footfall Dataset
def simulate_visitors(ts):
    """Simulate visitor count based on time of day and day of the week."""
    base = np.random.randint(10, 50)  # Base visitors
    if ts.dayofweek >= 5:  # Increase for weekends
        base += 30
    if 17 <= ts.hour <= 20:  # Increase for peak hours
        base += 40
    return base + np.random.randint(-10, 10)  # Add some noise

# Create a date range for July 2025 (changed from June 2024)
date_range = pd.date_range(start="2025-07-01", end="2025-07-31 23:00", freq='H')
df = pd.DataFrame({'timestamp': date_range})
df['visitors'] = df['timestamp'].apply(simulate_visitors)

# Save the simulated data to a CSV file
df.to_csv("footfall_data.csv", index=False)

# Step 2: Load and Preprocess Data
df = pd.read_csv("footfall_data.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)  # 1 if weekend, else 0
df['is_peak_hour'] = ((df['hour'] >= 17) & (df['hour'] <= 20)).astype(int)  # 1 if peak hour, else 0

# Step 3: Define Features and Target
X = df[['hour', 'day_of_week', 'is_weekend', 'is_peak_hour']]
y = df['visitors']

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Step 6: Predict & Evaluate Random Forest Model
y_pred_rf = rf_model.predict(X_test)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
print(f"Random Forest RMSE: {rmse_rf:.2f}")

# Step 7: Visualize Random Forest Results
plt.figure(figsize=(10, 5))
plt.plot(y_test.values[:50], label='Actual', marker='o')
plt.plot(y_pred_rf[:50], label='Predicted (RF)', marker='x')
plt.title('Actual vs Predicted Footfall (Random Forest)')
plt.xlabel('Sample Index')
plt.ylabel('Number of Visitors')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 8: Train the Neural Network Model
nn_model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # Output layer for regression
])

nn_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the neural network
nn_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)

# Step 9: Predict & Evaluate Neural Network Model
y_pred_nn = nn_model.predict(X_test)
rmse_nn = np.sqrt(mean_squared_error(y_test, y_pred_nn))
print(f"Neural Network RMSE: {rmse_nn:.2f}")

# Step 10: Visualize Neural Network Results
plt.figure(figsize=(10, 5))
plt.plot(y_test.values[:50], label='Actual', marker='o')
plt.plot(y_pred_nn[:50], label='Predicted (NN)', marker='x')
plt.title('Actual vs Predicted Footfall (Neural Network)')
plt.xlabel('Sample Index')
plt.ylabel('Number of Visitors')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 11: Predict for Future Dates (changed to August 1st 2025)
future_dates = pd.date_range("2025-08-01", "2025-08-01 23:00", freq='H')
future_df = pd.DataFrame({'timestamp': future_dates})
future_df['hour'] = future_df['timestamp'].dt.hour
future_df['day_of_week'] = future_df['timestamp'].dt.dayofweek
future_df['is_weekend'] = (future_df['day_of_week'] >= 5).astype(int)
future_df['is_peak_hour'] = ((future_df['hour'] >= 17) & (future_df['hour'] <= 20)).astype(int)

# Predict future footfall using Random Forest
future_X = future_df[['hour', 'day_of_week', 'is_weekend', 'is_peak_hour']]
future_pred_rf = rf_model.predict(future_X)

# Predict future footfall using Neural Network
future_pred_nn = nn_model.predict(future_X)

# Plot future predictions
plt.figure(figsize=(10, 4))
plt.plot(future_df['timestamp'], future_pred_rf, label='Predicted Footfall (RF)', marker='o')
plt.plot(future_df['timestamp'], future_pred_nn, label='Predicted Footfall (NN)', marker='x')
plt.title("Predicted Footfall for August 1st, 2025")  # Updated date in title
plt.xlabel("Hour")
plt.ylabel("Number of Visitors")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()