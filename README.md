# Ex.No: 6               HOLT WINTERS METHOD
### Date: 25-05-2026

### AIM:
To analyze and forecast tea vs coffee global using the Holt-Winters Time Series Forecasting Method by preprocessing the dataset, identifying trend and seasonal patterns, evaluating model performance using RMSE, and predicting future sales values.
### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

data = pd.read_csv('/content/drive/MyDrive/Time_Series/tea_vs_coffee_global_final.csv', parse_dates=['year'],index_col='year')

data.head()

# Select only the relevant numerical columns for time series analysis
numeric_cols = ['cups_per_day', 'monthly_spend']
data_monthly = data[numeric_cols].resample('MS').sum().fillna(0)

display(data_monthly.head())
data_monthly.plot(subplots=True, figsize=(10, 5), title='Monthly Sum of Cups Per Day and Monthly Spend')
plt.tight_layout()
plt.show()

scaler = MinMaxScaler()
time_series_to_decompose = data_monthly['monthly_spend'].to_frame()
scaled_series_values = scaler.fit_transform(time_series_to_decompose.values).flatten()
scaled_data = pd.Series(scaled_series_values, index=data_monthly.index)

scaled_data.plot()
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(scaled_data, model="additive", period=4)
decomposition.plot()
plt.show()

scaled_data=scaled_data+1 # multiplicative seasonality cant handle non postive values, ye
train_data = scaled_data[:int(len(scaled_data) * 0.8)]
test_data = scaled_data[int(len(scaled_data) * 0.8):]
model_add = ExponentialSmoothing(train_data, trend='add', seasonal='mul').fit()
test_predictions_add = model_add.forecast(steps=len(test_data))
ax=train_data.plot()
test_predictions_add.plot(ax=ax)
test_data.plot(ax=ax)
ax.legend(["Scaled Monthly Spend (Train)", "Scaled Monthly Spend (Predictions)","Scaled Monthly Spend (Test)"])
ax.set_title('Visual Evaluation of Scaled Monthly Spend Forecast')
ax.set_xlabel('Year')
ax.set_ylabel('Scaled Monthly Spend')
np.sqrt(mean_squared_error(test_data, test_predictions_add))
np.sqrt(scaled_data.var()),scaled_data.mean()

positive_cups_per_day = data_monthly['cups_per_day'] + 1

final_model = ExponentialSmoothing(positive_cups_per_day, trend='add', seasonal='add', seasonal_periods=12).fit()
final_predictions = final_model.forecast(steps=12) # Forecast for a full year

plt.figure(figsize=(10, 5))

# Full historical data with predictions
ax = data_monthly['cups_per_day'].plot(label='Cups Per Day (Actual)')
(final_predictions - 1).plot(ax=ax, label='Cups Per Day (Predictions)', linestyle='--')
ax.legend()
ax.set_xlabel('Year')
ax.set_ylabel('Cups Per Day')
ax.set_title('Cups Per Day Prediction (Full Range)')

plt.tight_layout()
plt.show()
```
### OUTPUT:
<img width="1337" height="627" alt="image" src="https://github.com/user-attachments/assets/3b8414d5-e5af-495b-998e-4eebacf98e85" />
<img width="844" height="592" alt="image" src="https://github.com/user-attachments/assets/d6ebfa19-c6ed-40ef-ae4c-24545d8fef97" />

<img width="798" height="566" alt="image" src="https://github.com/user-attachments/assets/51b1813a-789c-44ef-aaf4-6d179048f100" />

<img width="1298" height="601" alt="image" src="https://github.com/user-attachments/assets/c727c78d-9bec-4412-ae73-ffba9e48bfaa" />


### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
