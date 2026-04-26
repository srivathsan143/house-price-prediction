import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Sample dataset
data = pd.DataFrame({
    'area': [1000,1500,2000,2500,3000],
    'bedrooms': [2,3,3,4,4],
    'price': [200000,300000,400000,500000,600000]
})

X = data[['area','bedrooms']]
y = data['price']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
joblib.dump(model, 'model.pkl')

print("Model trained successfully!")
