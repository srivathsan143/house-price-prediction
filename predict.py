import joblib

model = joblib.load('model.pkl')

# Example prediction
result = model.predict([[2200, 3]])

print("Predicted Price:", result[0])
