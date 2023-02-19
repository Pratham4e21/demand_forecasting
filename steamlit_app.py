import pandas as pd
from flask import Flask, jsonify, request
from statsmodels.tsa.arima.model import ARIMA

app = Flask(__name__)

# Load the dataset into a pandas DataFrame
demand_data = pd.read_csv('demand_data.csv', index_col='Date', parse_dates=True)

# Clean and preprocess the data as needed

# Choose the ARIMA model to perform demand forecasting
model = ARIMA(demand_data, order=(1, 0, 0))
model_fit = model.fit()

@app.route('/forecast', methods=['GET'])
def forecast():
    # Get the input date from the user
    input_date = pd.to_datetime(request.args.get('date'))

    # Use the ARIMA model to make a forecast for the input date
    forecast = model_fit.forecast(steps=1)

    # Format the forecast as a JSON object and return it to the user
    result = {'date': input_date.strftime('%Y-%m-%d'), 'demand': forecast[0][0]}
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
