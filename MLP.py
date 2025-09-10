import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# 1. Load Dataset
df = pd.read_csv("https://raw.githubusercontent.com/funnyPhani/HouseData/main/kc_house_data.csv")

# Drop irrelevant columns
df = df.drop(["id", "date"], axis=1)

# One-hot encode zipcode if categorical
if df["zipcode"].dtype == "object":
    df = pd.get_dummies(df, columns=["zipcode"], drop_first=True)

# Features (X) and Target (y)
X = df.drop("price", axis=1)
y = df["price"]

# 2. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Build MLP Model
model = Sequential()
model.add(Dense(128, activation="relu", input_shape=(X_train.shape[1],)))
model.add(Dropout(0.3))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(32, activation="relu"))
model.add(Dense(1))  

# Compile Model
model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])

# 5. Train Model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    verbose=1
)

# 6. Evaluate Model
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Mean Absolute Error: {mae:.2f}")

# 7. Example Prediction
y_pred = model.predict(X_test[:5])
print("\nPredicted Prices:", y_pred.flatten())
print("Actual Prices   :", y_test.iloc[:5].values)
