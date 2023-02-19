import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from flask import Flask, request, jsonify

# Load the dataset into a pandas DataFrame
df = pd.read_csv('sales_data.csv', parse_dates=['Date'])

# Clean and preprocess the data as needed

# Define the features and target variables
X = df[['Year', 'Month', 'Weekday']]
y = df['Sales']

# Train a linear regression model on the historical data
model = LinearRegression()
model.fit(X, y)

# Define a Flask app
app = Flask(__name__)

# Define a route to handle incoming requests for demand forecasting
@app.route('/forecast', methods=['POST'])
def forecast():
    # Parse the incoming JSON data
    data = request.get_json()
    year, month, weekday = data['year'], data['month'], data['weekday']
    
    # Make a demand forecast using the trained linear regression model
    forecast = model.predict([[year, month, weekday]])
    
    # Return the forecast as a JSON response
    return jsonify({'forecast': int(forecast[0])})

# Define a route to display a chart of the historical demand data
@app.route('/')
def chart():
    fig, ax = plt.subplots()
    ax.plot(df['Date'], df['Sales'])
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales')
    fig.autofmt_xdate()
    plt.show()
    
if __name__ == '__main__':
    app.run()