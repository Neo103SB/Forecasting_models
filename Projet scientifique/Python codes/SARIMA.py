import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load the Superstore sales dataset into a Pandas DataFrame
df = pd.read_csv("sales_data.csv")

# Select only the rows for the furniture category
df = df[df["Category"] == "Furniture"]

# Convert the Order Date column to a datetime data type and set it as the index of the DataFrame
df["Order Date"] = pd.to_datetime(df["Order Date"])
df.set_index("Order Date", inplace=True)

# Resample the data to a monthly frequency, taking the sum of all daily sales values in each month
monthly_sales = (df["Sales"].resample("M").sum())/30
# Split the data into training and test sets
train = monthly_sales.iloc[:36]
test = monthly_sales.iloc[36:]

# Fit the SARIMA model to the training data
model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
model_fit = model.fit()

# Make predictions on the test set
predictions = model_fit.predict(start=test.index[0], end=test.index[-1])

# Evaluate the model's performance
mse = ((predictions - test) ** 2).mean()
rmse = mse ** 0.5
mape = (abs(predictions - test) / test).mean()
print("MSE: ", mse)
print("RMSE: ", rmse)
print("MAPE: ", mape)

# Import matplotlib
import matplotlib.pyplot as plt

# Plot the time-series data
plt.plot(monthly_sales, label='Real sales')

# Plot the predictions
plt.plot(predictions[1:12], label='Predictions')

# Add a legend to the plot
plt.legend()

# Add x- and y-axis labels
plt.xlabel('Time')
plt.ylabel('Sales')

# Show the plot
plt.show()
