# all libraries needed
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error

"""
1. Covid-19 general data
-----------------------
"""

# test positivity and deaths by month
# File path
file_path = 'caseNumber.csv'
# Read the file
case = pd.read_csv(file_path)    # case for covid case analysis

case = case.dropna(subset=["Year"])
case = case.dropna(subset=["Month"])
case = case.dropna(subset=["Day"])

case["Year"] = case["Year"].astype(int)
case["Month"] = case["Month"].astype(int)
case["Day"] = case["Day"].astype(int)

# Group by Year and Quarter and compute both Deaths sum and Test Positivity average
case_data = case.groupby(['Year', 'Month']).agg(
    {'Deaths': 'sum', 'Test Positivity(%)': 'mean'}).reset_index()



# Create a new column "Time" for plotting
case_data["Time"] = case_data['Year'].astype(str) + " " + case_data['Month'].astype(str)


# Plotting
fig, ax1 = plt.subplots(figsize=(18, 9))

# Plot for Deaths (on left axis)
ax1.plot(case_data["Time"], case_data["Deaths"], marker='o', color='b', linestyle='-', linewidth=2, markersize=6, label="Deaths")
ax1.set_xlabel('Time Line', fontsize=10)
ax1.set_ylabel('Total Deaths', fontsize=12, color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.set_xticklabels(case_data["Time"], rotation=45, fontsize=12)

# Create a second y-axis for Test Positivity (on right axis)
ax2 = ax1.twinx()
ax2.plot(case_data["Time"], case_data["Test Positivity(%)"], marker='o', color='r', linestyle='-.', linewidth=2, markersize=6, label="Test Positivity(%)")
ax2.set_ylabel('Test Positivity(%)', fontsize=12, color='black')
ax2.tick_params(axis='y', labelcolor='black')

# Titles and grid
plt.title('Total Deaths and Test Positivity by Month', fontsize=14)
ax1.grid(True)

# Show plot
fig.tight_layout()
plt.show()

# test positivity by quarter
plt.figure(figsize=(12, 6))


quarter_case_data = case.groupby(['Year','Quarter'])['Test Positivity(%)'].mean().reset_index()

quarter_case_data["Time"] = quarter_case_data["Year"].astype(str) + " " + quarter_case_data["Quarter"]



"""
2. economy part
------------------
"""

# plot real GDP and test positivity
# Load economics data
file = 'economics.csv'
econ = pd.read_csv(file)

# Create a 'Time' column in the 'econ' DataFrame
econ["Time"] = econ["Year"].astype(str) + " " + econ["Quarter"]

# Filter 'econ' data to match the range of quarter_case_data
econ_filtered = econ[(econ["Time"] >= "2019 Q4") & (econ["Time"] <= "2024 Q3")]


# Create a plot with the first y-axis for Real GDP
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot Real GDP with 'econ_filtered' for consistency
ax1.plot(econ_filtered["Time"], econ_filtered["Real GDP"], marker='o', label="Real GDP", color='b')
ax1.set_xlabel("Time Line", fontsize=12)
ax1.set_ylabel("Real GDP", fontsize=12, color='b')
ax1.tick_params(axis='y', labelcolor='b')

# Set the x-ticks to use the 'Time' from filtered 'econ' data
ax1.set_xticklabels(econ_filtered["Time"], rotation=45, fontsize=10)

# Create a second y-axis for Test Positivity
ax2 = ax1.twinx()

# Plot Test Positivity, ensuring we filter quarter_case_data to the same time range
ax2.plot(quarter_case_data["Time"], quarter_case_data["Test Positivity(%)"], marker='o', color='r', linestyle='-.',
         linewidth=2, markersize=6, label="Test Positivity(%)")
ax2.set_ylabel("Test Positivity(%)", fontsize=12, color='r')
ax2.tick_params(axis='y', labelcolor='r')

# Add title, grid, and formatting
plt.title("Real GDP and Test Positivity by Quarters", fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)

# Adjust layout
fig.tight_layout()

# Show legend
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")

# Display plot
plt.show()


# Plotting total export,import,net exports
ax1 = plt.gca()
ax1.plot(econ_filtered["Time"], econ_filtered['Total Exports'], marker='o', color='blue',label='Total Exports', linestyle='--')
ax1.plot(econ_filtered["Time"], econ_filtered['Total Imports'], marker='o', color='red',label='Total Imports')
ax1.plot(econ_filtered["Time"], econ_filtered['Net Exports'], marker='o', color='black',label='Net Exports')

# Set labels and formatting for the first axis
ax1.set_xlabel('Time Line', fontsize=12)
ax1.set_ylabel('Value (in Billion)', fontsize=12)
ax1.tick_params(axis='y', labelsize=10)
ax1.grid(True, linestyle='--', alpha=0.7)

plt.xticks(econ_filtered["Time"], rotation=45, fontsize=10)

# Create the second y-axis
ax2 = ax1.twinx()
ax2.plot(quarter_case_data["Time"], quarter_case_data["Test Positivity(%)"], marker='o', color='green', linestyle='-.',
         linewidth=2, markersize=6, label="Test Positivity(%)")
ax2.set_ylabel("Test Positivity(%)", fontsize=12, color='green')
ax2.tick_params(axis='y', labelcolor='green', labelsize=10)

# Add a title
plt.title('Exports, Imports, and Test Positivity Rate', fontsize=14)

# Combine legends from both axes
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=10)

# Ensure the layout fits well
plt.tight_layout()

# Show the plot
plt.show()


#Predicting using Arima

#using data from year 2014 to year 2022 for prediction
econ_test = econ[(econ["Time"] <= "2022 Q3")]

exports = econ_test["Total Exports"].dropna()
imports = econ_test["Total Imports"].dropna()
netExports = econ_test["Net Exports"].dropna()
max_lags = min(30, len(exports) // 2)

# ACF for Exports
fig, ax = plt.subplots(2, 1, figsize=(10, 7))
plot_acf(exports, lags=max_lags, ax=ax[0])
ax[0].set_title("Exports ACF")

# PACF for Exports
plot_pacf(exports.dropna(), lags=max_lags, ax=ax[1])
ax[1].set_title("Exports PACF")


# ACF for imports
fig, ax = plt.subplots(2, 1, figsize=(10, 7))
plot_acf(imports, lags=max_lags, ax=ax[0])
ax[0].set_title("Imports ACF")

# PACF for imports
plot_pacf(imports, lags=max_lags, ax=ax[1])
ax[1].set_title("Imports PACF")

# ACF for Net Exports
fig, ax = plt.subplots(2, 1, figsize=(10, 7))
plot_acf(netExports, lags=max_lags, ax=ax[0])
ax[0].set_title("Net Exports ACF")

# PACF for Net Exports
plot_pacf(netExports, lags=max_lags, ax=ax[1])
ax[1].set_title("Net Exports PACF")

# ARIMA model (adjust p, d, q based on ACF/PACF)
model = ARIMA(exports, order=(3, 2, 7))
model_fit = model.fit()

# Forecasting the next 5 years
forecast_exports = model_fit.forecast(steps=20)


# Plot actual vs forecast
plt.figure(figsize=(10, 5))
plt.plot(econ["Time"], econ['Total Exports'], marker='o', color='blue',linestyle='--', label="Actual Exports")
plt.plot(forecast_exports.index, forecast_exports, label="Forecasted Exports", color="red")
plt.xticks(econ["Time"], rotation=45, fontsize=8)
plt.legend()
plt.title("Exports Forecast Comparison")
plt.show()


plt.tight_layout()
plt.show()


# imports model
model = ARIMA(imports, order=(5, 3, 9))
model_fit = model.fit()

# Forecasting the next 5 years
forecast_imports = model_fit.forecast(steps=20)

# Plot actual vs forecast
plt.figure(figsize=(10, 5))
plt.plot(econ["Time"], econ['Total Imports'], marker='o', color='blue',linestyle='--', label="Actual Imports")
plt.plot(forecast_imports.index, forecast_imports, label="Forecasted Imports", color="red")
plt.xticks(econ["Time"], rotation=45, fontsize=8)
plt.legend()
plt.title("imports Forecast Comparison")

plt.tight_layout()
plt.show()


# NetExports model
model = ARIMA(netExports, order=(7, 2, 19))
model_fit = model.fit()

# Forecasting the next 5 years
forecast_Nx = model_fit.forecast(steps=20)

# Plot actual vs forecast
plt.figure(figsize=(10, 5))
plt.plot(econ["Time"], econ['Net Exports'], marker='o', color='blue',linestyle='--', label="Actual Imports")
plt.plot(forecast_Nx.index, forecast_Nx, label="Forecasted Net Exports", color="red")
plt.xticks(econ["Time"], rotation=45, fontsize=8)
plt.legend()
plt.title("Net Exports Forecast Comparison")

plt.tight_layout()
plt.show()


"""
3.Travel 
---------------------
"""



"""
4. Remote Work
----------------------------
"""
# Total Work
# Load the dataset
file_path = 'remote_work.csv'
df = pd.read_csv(file_path)

# Extract and clean the relevant columns
df['Date'] = pd.to_datetime(df['Date'])  # Convert date to datetime
time_series_data = df[['Date', 'Total_Adjusted']].dropna()
time_series_data.set_index('Date', inplace=True)

# Plot the data to visualize
plt.figure(figsize=(10, 6))
plt.plot(time_series_data, label="Total Remote Work", marker='o')
plt.plot(df['Date'], df['Fully_Remote'], label="Completely Remote Work", marker='o')
plt.plot(df['Date'], df['Some_Remote'], label="Partially Remote Work", marker='o')
plt.title("Combined Percent of Employed Individuals Who Did Remote Work Over Time")
plt.xlabel("Date")
plt.ylabel("Total Remote Work (%)")
plt.legend()
plt.grid()
plt.show()


# stationary Test
adf_test = adfuller(time_series_data['Total_Adjusted'])
print(f"ADF Statistic: {adf_test[0]}")
print(f"p-value: {adf_test[1]}")
print("Stationary" if adf_test[1] < 0.05 else "Non-Stationary")


plot_acf(time_series_data['Total_Adjusted'], lags=20)
plot_pacf(time_series_data['Total_Adjusted'], lags=20)


# Fit the ARIMA model
model = ARIMA(time_series_data['Total_Adjusted'], order=(2, 0, 16))
model_fit = model.fit()

# Summarize the results
print(model_fit.summary())


# Forecast the next 4 years
forecast = model_fit.get_forecast(steps=48)
forecast_index = pd.date_range(
    start=time_series_data.index[-1], periods=48, freq='ME'
)
forecast_values = forecast.predicted_mean

# Plot the forecast
plt.figure(figsize=(12, 6))
plt.plot(time_series_data['Total_Adjusted'], label="Historical Data")
plt.plot(forecast_index, forecast_values, label="Forecast", color="orange")
plt.title("ARIMA Forecast for Combined Remote Work")
plt.xlabel("Date")
plt.ylabel("Total Remote Work (%)")
plt.legend()
plt.grid()
plt.show()


# Mean Squared Error calculation
predicted = model_fit.predict(start=0, end=len(time_series_data) - 1)
mse = mean_squared_error(time_series_data['Total_Adjusted'], predicted)
print(f"Mean Squared Error: {mse}")


# Full Time Remote Work Only
# Extract and clean the relevant columns
df['Date'] = pd.to_datetime(df['Date'])  # Convert date to datetime
time_series_data = df[['Date', 'FT_Total_Adjusted']].dropna()
time_series_data.set_index('Date', inplace=True)

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(time_series_data, label="Total Remote Work", marker='o')
plt.plot(df['Date'], df['FT_Fully_Remote'], label="Completely Remote Work", marker='o')
plt.plot(df['Date'], df['FT_Some_Remote'], label="Partially Remote Work", marker='o')
plt.title("Percent of Full-Time Employed Individuals Who Did Remote Work Over Time")
plt.xlabel("Date")
plt.ylabel("Total Remote Work (%)")
plt.legend()
plt.grid()
plt.show()


adf_test = adfuller(time_series_data['FT_Total_Adjusted'])
print(f"ADF Statistic: {adf_test[0]}")
print(f"p-value: {adf_test[1]}")
print("Stationary" if adf_test[1] < 0.05 else "Non-Stationary")


plot_acf(time_series_data['FT_Total_Adjusted'], lags=20)
plot_pacf(time_series_data['FT_Total_Adjusted'], lags=20)


# Fit the ARIMA model
model = ARIMA(time_series_data['FT_Total_Adjusted'], order=(2, 0, 16))
model_fit = model.fit()

# Summarize the results
print(model_fit.summary())

# Forecast the next 4 years
forecast = model_fit.get_forecast(steps=48)
forecast_index = pd.date_range(
    start=time_series_data.index[-1], periods=48, freq='ME'
)
forecast_values = forecast.predicted_mean

# Plot the forecast
plt.figure(figsize=(12, 6))
plt.plot(time_series_data['FT_Total_Adjusted'], label="Historical Data")
plt.plot(forecast_index, forecast_values, label="Forecast", color="orange")
plt.title("ARIMA Forecast for Full-Time Remote Work")
plt.xlabel("Date")
plt.ylabel("Total Remote Work (%)")
plt.legend()
plt.grid()
plt.show()

# Mean Squared Error calculation
predicted = model_fit.predict(start=0, end=len(time_series_data) - 1)
mse = mean_squared_error(time_series_data['FT_Total_Adjusted'], predicted)
print(f"Mean Squared Error: {mse}")

# Part Time Remote Work Only
# Extract and clean the relevant columns
df['Date'] = pd.to_datetime(df['Date'])  # Convert date to datetime
time_series_data = df[['Date', 'PT_Total_Adjusted']].dropna()
time_series_data.set_index('Date', inplace=True)

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(time_series_data, label="Total Remote Work", marker='o')
plt.plot(df['Date'], df['PT_Fully_Remote'], label="Completely Remote Work", marker='o')
plt.plot(df['Date'], df['PT_Some_Remote'], label="Partially Remote Work", marker='o')
plt.title("Percent of Part-Time Employed Individuals Who Did Remote Work Over Time")
plt.xlabel("Date")
plt.ylabel("Total Remote Work (%)")
plt.legend()
plt.grid()
plt.show()

adf_test = adfuller(time_series_data['PT_Total_Adjusted'])
print(f"ADF Statistic: {adf_test[0]}")
print(f"p-value: {adf_test[1]}")
print("Stationary" if adf_test[1] < 0.05 else "Non-Stationary")

plot_acf(time_series_data['PT_Total_Adjusted'], lags=20)
plot_pacf(time_series_data['PT_Total_Adjusted'], lags=20)

# Fit the ARIMA model
model = ARIMA(time_series_data['PT_Total_Adjusted'], order=(2, 0, 14))
model_fit = model.fit()

# Summarize the results
print(model_fit.summary())

# Forecast the next 4 years
forecast = model_fit.get_forecast(steps=48)
forecast_index = pd.date_range(
    start=time_series_data.index[-1], periods=48, freq='ME'
)
forecast_values = forecast.predicted_mean

# Plot the forecast
plt.figure(figsize=(12, 6))
plt.plot(time_series_data['PT_Total_Adjusted'], label="Historical Data")
plt.plot(forecast_index, forecast_values, label="Forecast", color="orange")
plt.title("ARIMA Forecast for Part-Time Remote Work")
plt.xlabel("Date")
plt.ylabel("Total Remote Work (%)")
plt.legend()
plt.grid()
plt.show()

# Mean Squared Error calculation
predicted = model_fit.predict(start=0, end=len(time_series_data) - 1)
mse = mean_squared_error(time_series_data['PT_Total_Adjusted'], predicted)
print(f"Mean Squared Error: {mse}")

"""
5. Mental Health
----------------------------
"""

# NATIONAL AVERAGE
df = pd.read_csv('anxiety_depression.csv')

# Filter for the Indicator "Symptoms of Anxiety Disorder or Depressive Disorder"
df_filtered = df[df['Indicator'] == 'Symptoms of Anxiety Disorder or Depressive Disorder']

# Filter for the Group "National Estimate"
df_filtered = df_filtered[df_filtered['Group'] == 'National Estimate']

# Filter for Time Periods 1 to 72
df_filtered = df_filtered[(df_filtered['Time Period'] >= 1) & (df_filtered['Time Period'] <= 72)]

#  Handle duplicates by grouping by Time Period and calculating the mean for each time period
df_filtered = df_filtered.groupby('Time Period').agg({'Value': 'mean', 'Time Period End Date': 'first'}).reset_index()

#  Convert 'Time Period End Date' to datetime format
df_filtered['Time Period End Date'] = pd.to_datetime(df_filtered['Time Period End Date'])

# Set 'Time Period End Date' as the index (using original date range)
df_filtered.set_index('Time Period End Date', inplace=True)

# Resample the time series to a regular frequency (e.g., monthly) using interpolation
df_filtered_resampled = df_filtered.resample('ME').mean()  # Resampling to monthly frequency, using the mean
df_filtered_resampled['Value'] = df_filtered_resampled['Value'].interpolate(method='linear')  # Interpolating missing values

# Apply Holt-Winters Exponential Smoothing to the resampled data
model_hw = ExponentialSmoothing(
    df_filtered_resampled['Value'],
    trend='add',  # You can also try 'mul' for multiplicative trend
    seasonal='add',  # Use 'mul' for multiplicative seasonality if data suggests it
    seasonal_periods=12  # Adjust this based on your data, e.g., if your data shows yearly seasonality
)

# Fit the model
hw_fit = model_hw.fit()


#  Forecast future values (e.g., the next 12 periods)
future_periods = pd.date_range(df_filtered_resampled.index[-1], periods=24, freq='ME')[1:]  # Forecasting the next 12 periods
forecast_values_hw = hw_fit.forecast(len(future_periods))

# Set the forecasted future periods as the index for the forecasted values
forecast_df = pd.DataFrame(forecast_values_hw, index=future_periods, columns=['Forecast'])



# Combine historical fitted values and forecasted values using pd.concat()
combined_actuals = pd.concat([df_filtered_resampled['Value'], pd.Series([np.nan] * len(future_periods), index=future_periods)])
combined_fitted_forecasted = pd.concat([hw_fit.fittedvalues, pd.Series(forecast_values_hw, index=future_periods)])

# Calculate SS_residual (sum of squared residuals) for the entire series
ss_residual_combined = np.sum((combined_actuals - combined_fitted_forecasted) ** 2)

# Calculate SS_total (total sum of squares) for the entire series
ss_total_combined = np.sum((combined_actuals.dropna() - combined_actuals.dropna().mean()) ** 2)

# Calculate R^2
r_squared_combined = 1 - (ss_residual_combined / ss_total_combined)

# Print R^2
print(f"R^2 for combined (fitted + forecasted) values: {r_squared_combined}")



# Plot the results
plt.figure(figsize=(12, 6))

# Plot the observed data (original data points)
plt.plot(df_filtered_resampled.index, df_filtered_resampled['Value'], label='Observed Data', color='blue')

# Plot the fitted model (Holt-Winters smoothing)
plt.plot(df_filtered_resampled.index, hw_fit.fittedvalues, label='Fitted Model (Holt-Winters)', color='blue', linestyle=':')

# Plot the predicted future values
plt.plot(forecast_df.index, forecast_df['Forecast'], label='Predicted Future Values', color='red', linestyle='--')
observed_end_date = df_filtered_resampled.index[-1]
plt.axvline(x=observed_end_date, color='grey', linewidth = 2.5, linestyle='--', label='Start of Prediction')
plt.axhline(y=11, color='black', linestyle='-', linewidth=2, label='2019 Threshold')
# Adding labels and title
plt.xlabel('Date')
plt.ylabel('Percent of Adults with Symptoms')
plt.title('National Average Prediction of Symptoms of Anxiety Disorder or Depressive Disorder Using Holt-Winters Exponential Smoothing')
plt.legend()

# Display the plot
plt.tight_layout()
plt.show()



# MALE AND FEMALE


df = pd.read_csv('anxiety_depression.csv')

# Filter for the Indicator "Symptoms of Anxiety Disorder or Depressive Disorder"
df_filtered = df[df['Indicator'] == 'Symptoms of Anxiety Disorder or Depressive Disorder']

# Filter for the Group "By Sex"
df_filtered = df_filtered[df_filtered['Group'] == 'By Sex']

# Function to apply Holt-Winters for a specific subgroup (e.g., Male or Female)
def apply_holt_winters_for_subgroup(subgroup_name):
    # Filter by subgroup (e.g., Male or Female)
    df_subgroup = df_filtered[df_filtered['Subgroup'] == subgroup_name]

    # Filter for Time Periods 1 to 72
    df_subgroup = df_subgroup[(df_subgroup['Time Period'] >= 1) & (df_subgroup['Time Period'] <= 72)]

    #  Handle duplicates by grouping by Time Period and calculating the mean for each time period
    df_subgroup = df_subgroup.groupby('Time Period').agg({'Value': 'mean', 'Time Period End Date': 'first'}).reset_index()

    # Convert 'Time Period End Date' to datetime format
    df_subgroup['Time Period End Date'] = pd.to_datetime(df_subgroup['Time Period End Date'])

    #  Set 'Time Period End Date' as the index (using original date range)
    df_subgroup.set_index('Time Period End Date', inplace=True)

    # Resample the time series to a regular frequency (e.g., monthly) using interpolation
    df_subgroup_resampled = df_subgroup.resample('ME').mean()  # Resampling to monthly frequency, using the mean
    df_subgroup_resampled['Value'] = df_subgroup_resampled['Value'].interpolate(method='linear')  # Interpolating missing values

    #  Apply Holt-Winters Exponential Smoothing to the resampled data
    model_hw = ExponentialSmoothing(
        df_subgroup_resampled['Value'],
        trend='add',  # You can also try 'mul' for multiplicative trend
        seasonal='add',  # Use 'mul' for multiplicative seasonality if data suggests it
        seasonal_periods=12  # Adjust this based on your data, e.g., if your data shows yearly seasonality
    )

    # Fit the model
    hw_fit = model_hw.fit()


    # Forecast future values (e.g., the next 12 periods)
    future_periods = pd.date_range(df_subgroup_resampled.index[-1], periods=24, freq='ME')[1:]  # Forecasting the next 12 periods
    forecast_values_hw = hw_fit.forecast(len(future_periods))

    # Set the forecasted future periods as the index for the forecasted values
    forecast_df = pd.DataFrame(forecast_values_hw, index=future_periods, columns=['Forecast'])

    return df_subgroup_resampled, hw_fit, forecast_df

# Function to calculate R^2 for combined historical and forecasted data
def calculate_r_squared(df_resampled, hw_fit, forecast_values_hw, future_periods):
    # Combine historical fitted values and forecasted values using pd.concat()
    combined_actuals = pd.concat([df_resampled['Value'], pd.Series([np.nan] * len(future_periods), index=future_periods)])
    combined_fitted_forecasted = pd.concat([hw_fit.fittedvalues, pd.Series(forecast_values_hw, index=future_periods)])

    # Calculate SS_residual (sum of squared residuals) for the entire series
    ss_residual_combined = np.sum((combined_actuals - combined_fitted_forecasted) ** 2)

    # Calculate SS_total (total sum of squares) for the entire series
    ss_total_combined = np.sum((combined_actuals.dropna() - combined_actuals.dropna().mean()) ** 2)

    # Calculate R^2
    r_squared_combined = 1 - (ss_residual_combined / ss_total_combined)

    return r_squared_combined
# Apply Holt-Winters for Male
df_male_resampled, hw_male_fit, forecast_male_df = apply_holt_winters_for_subgroup('Male')

# Apply Holt-Winters for Female
df_female_resampled, hw_female_fit, forecast_female_df = apply_holt_winters_for_subgroup('Female')


r_squared_female = calculate_r_squared(df_female_resampled, hw_female_fit, forecast_female_df['Forecast'], forecast_female_df.index)

# Calculate R^2 for Male (combined historical + forecasted)
r_squared_male = calculate_r_squared(df_male_resampled, hw_male_fit, forecast_male_df['Forecast'], forecast_male_df.index)

# Print the R^2 results for both groups
print(f"Female R^2: {r_squared_female}")
print(f"Male R^2: {r_squared_male}")


# Plot the results
plt.figure(figsize=(12, 6))

# Plot the observed data (original data points) for Male
plt.plot(df_male_resampled.index, df_male_resampled['Value'], label='Observed Data (Male)', color='blue')

# Plot the fitted model (Holt-Winters smoothing) for Male
plt.plot(df_male_resampled.index, hw_male_fit.fittedvalues, label='Fitted Model (Male)', color='blue', linestyle=':')

# Plot the predicted future values for Male
plt.plot(forecast_male_df.index, forecast_male_df['Forecast'], label='Predicted Future Values (Male)', color='blue', linestyle='--')

# Plot the observed data (original data points) for Female
plt.plot(df_female_resampled.index, df_female_resampled['Value'], label='Observed Data (Female)', color='green')

# Plot the fitted model (Holt-Winters smoothing) for Female
plt.plot(df_female_resampled.index, hw_female_fit.fittedvalues, label='Fitted Model (Female)', color='green', linestyle=':')

# Plot the predicted future values for Female
plt.plot(forecast_female_df.index, forecast_female_df['Forecast'], label='Predicted Future Values (Female)', color='green', linestyle='--')
observed_end_date_male = df_male_resampled.index[-1]
plt.axvline(x=observed_end_date_male, color='grey', linewidth = 2.5, linestyle='--', label='Start of Prediction')
plt.axhline(y=11, color='black', linestyle='-', linewidth=2, label='2019 Threshold')
# Adding labels and title
plt.xlabel('Date')
plt.ylabel('Percent of Adults with Symptoms')
plt.title('Prediction of Symptoms of Anxiety Disorder or Depressive Disorder by Sex Using Holt-Winters Exponential Smoothing')
plt.legend()

# Display the plot
plt.tight_layout()
plt.show()



#REPUBLICAN VS DEMOCRAT


df = pd.read_csv('anxiety_depression.csv')

# Filter for the Indicator "Symptoms of Anxiety Disorder or Depressive Disorder"
df_filtered = df[df['Indicator'] == 'Symptoms of Anxiety Disorder or Depressive Disorder']

# Filter for the Group "By State"
df_filtered = df_filtered[df_filtered['Group'] == 'By State']

republican = ['Florida', 'Alabama', 'Mississippi', 'Louisiana', 'Arkansas', 'Texas', 'Oklahoma',
              'Kansas', 'Missouri', 'Iowa', 'Nebraska', 'South Dakota', 'North Dakota', 'Wyoming', 'Montana',
              'Idaho', 'Alaska', 'Tennessee', 'South Carolina', 'North Carolina', 'Kentucky', 'West Virginia', 'Ohio', 'Indiana', 'Utah']
democrat = ['Georgia', 'New Mexico', 'Colorado', 'Arizona', 'Nevada', 'California', 'Oregon', 'Washington',
            'Minnesota', 'Wisconsin', 'Illinois', 'Michigan', 'Pennsylvania', 'Virginia',
            'Maryland', 'Delaware', 'New Jersey', 'New York', 'Rhode Island', 'Connecticut', 'Massachusetts', 'Vermont',
            'New Hampshire', 'Maine', 'Hawaii', 'District of Columbia']
# Function to apply Holt-Winters for a specific group of states (e.g., Republican or Democrat)
def apply_holt_winters_for_state_group(state_group):
    # Filter by states in the group (Republican or Democrat)
    df_state_group = df_filtered[df_filtered['Subgroup'].isin(state_group)]

    # Filter for Time Periods 1 to 72
    df_state_group = df_state_group[(df_state_group['Time Period'] >= 1) & (df_state_group['Time Period'] <= 72)]

    #  Handle duplicates by grouping by Time Period and calculating the mean for each time period
    df_state_group = df_state_group.groupby('Time Period').agg({'Value': 'mean', 'Time Period End Date': 'first'}).reset_index()

    # Convert 'Time Period End Date' to datetime format
    df_state_group['Time Period End Date'] = pd.to_datetime(df_state_group['Time Period End Date'])

    #  Set 'Time Period End Date' as the index (using original date range)
    df_state_group.set_index('Time Period End Date', inplace=True)

    # Resample the time series to a regular frequency (e.g., monthly) using interpolation
    df_state_group_resampled = df_state_group.resample('ME').mean()  # Resampling to monthly frequency, using the mean
    df_state_group_resampled['Value'] = df_state_group_resampled['Value'].interpolate(method='linear')  # Interpolating missing values

    # Apply Holt-Winters Exponential Smoothing to the resampled data
    model_hw = ExponentialSmoothing(
        df_state_group_resampled['Value'],
        trend='add',  # You can also try 'mul' for multiplicative trend
        seasonal='add',  # Use 'mul' for multiplicative seasonality if data suggests it
        seasonal_periods=12  # Adjust this based on your data, e.g., if your data shows yearly seasonality
    )

    # Fit the model
    hw_fit = model_hw.fit()

    #  Forecast future values (e.g., the next 12 periods)
    future_periods = pd.date_range(df_state_group_resampled.index[-1], periods=24, freq='ME')[1:]  # Forecasting the next 12 periods
    forecast_values_hw = hw_fit.forecast(len(future_periods))

    #  Set the forecasted future periods as the index for the forecasted values
    forecast_df = pd.DataFrame(forecast_values_hw, index=future_periods, columns=['Forecast'])

    return df_state_group_resampled, hw_fit, forecast_df
# Function to calculate R^2 for combined historical and forecasted data
def calculate_r_squared(df_resampled, hw_fit, forecast_values_hw, future_periods):
    # Combine historical fitted values and forecasted values using pd.concat()
    combined_actuals = pd.concat([df_resampled['Value'], pd.Series([np.nan] * len(future_periods), index=future_periods)])
    combined_fitted_forecasted = pd.concat([hw_fit.fittedvalues, pd.Series(forecast_values_hw, index=future_periods)])

    # Calculate SS_residual (sum of squared residuals) for the entire series
    ss_residual_combined = np.sum((combined_actuals - combined_fitted_forecasted) ** 2)

    # Calculate SS_total (total sum of squares) for the entire series
    ss_total_combined = np.sum((combined_actuals.dropna() - combined_actuals.dropna().mean()) ** 2)

    # Calculate R^2
    r_squared_combined = 1 - (ss_residual_combined / ss_total_combined)

    return r_squared_combined

# Apply Holt-Winters for Republican states
df_republican_resampled, hw_republican_fit, forecast_republican_df = apply_holt_winters_for_state_group(republican)

# Apply Holt-Winters for Democrat states
df_democrat_resampled, hw_democrat_fit, forecast_democrat_df = apply_holt_winters_for_state_group(democrat)

# Calculate R^2 for Republican states (combined historical + forecasted)
r_squared_republican = calculate_r_squared(df_republican_resampled, hw_republican_fit, forecast_republican_df['Forecast'], forecast_republican_df.index)

# Calculate R^2 for Democrat states (combined historical + forecasted)
r_squared_democrat = calculate_r_squared(df_democrat_resampled, hw_democrat_fit, forecast_democrat_df['Forecast'], forecast_democrat_df.index)

# Print the R^2 results for both groups
print(f"Republican States R^2: {r_squared_republican}")
print(f"Democrat States R^2: {r_squared_democrat}")

#Plot the results
plt.figure(figsize=(14, 8))

# Plot the observed data (original data points) for Republican states
plt.plot(df_republican_resampled.index, df_republican_resampled['Value'], label='Observed Data (Republican States)', color='red')

# Plot the fitted model (Holt-Winters smoothing) for Republican states
plt.plot(df_republican_resampled.index, hw_republican_fit.fittedvalues, label='Fitted Model (Republican States)', color='red', linestyle = ':')

# Plot the predicted future values for Republican states
plt.plot(forecast_republican_df.index, forecast_republican_df['Forecast'], label='Predicted Future Values (Republican States)', color='red', linestyle='--')

# Plot the observed data (original data points) for Democrat states
plt.plot(df_democrat_resampled.index, df_democrat_resampled['Value'], label='Observed Data (Democrat States)', color='blue')

# Plot the fitted model (Holt-Winters smoothing) for Democrat states
plt.plot(df_democrat_resampled.index, hw_democrat_fit.fittedvalues, label='Fitted Model (Democrat States)', color='blue', linestyle = ':')

# Plot the predicted future values for Democrat states
plt.plot(forecast_democrat_df.index, forecast_democrat_df['Forecast'], label='Predicted Future Values (Democrat States)', color='blue', linestyle='--')
observed_end_date = df_republican_resampled.index[-1]  # Get the last date of observed data
plt.axvline(x=observed_end_date, color='grey', linewidth = 2.5, linestyle='--', label='Start of Prediction')
plt.axhline(y=11, color='black', linestyle='-', linewidth=2, label='2019 Threshold')
# Adding labels and title
plt.xlabel('Date')
plt.ylabel('Percent of Adults with Symptoms')
plt.title('Comparison of Symptoms of Anxiety Disorder or Depressive Disorder by State Political Affiliation Using Holt-Winters Exponential Smoothing')
plt.legend()

# Display the plot
plt.tight_layout()
plt.show()


## FOR EDUCATION LEVEL


df = pd.read_csv('anxiety_depression.csv')

# Filter for the Indicator "Symptoms of Anxiety Disorder or Depressive Disorder"
df_filtered = df[df['Indicator'] == 'Symptoms of Anxiety Disorder or Depressive Disorder']

# Filter for the Group "By Education"
df_filtered = df_filtered[df_filtered['Group'] == 'By Education']

#Define a function to apply Holt-Winters for a specific education level group
def apply_holt_winters_for_education_group(education_group):
    # Filter by education level
    df_education_group = df_filtered[df_filtered['Subgroup'] == education_group]

    # Filter for Time Periods 1 to 72
    df_education_group = df_education_group[(df_education_group['Time Period'] >= 1) & (df_education_group['Time Period'] <= 72)]

    df_education_group = df_education_group.groupby('Time Period').agg({'Value': 'mean', 'Time Period End Date': 'first'}).reset_index()

    df_education_group['Time Period End Date'] = pd.to_datetime(df_education_group['Time Period End Date'])

    df_education_group.set_index('Time Period End Date', inplace=True)

    #Resample the time series to a regular frequency (e.g., monthly) using interpolation
    df_education_group_resampled = df_education_group.resample('ME').mean()  # Resampling to monthly frequency, using the mean
    df_education_group_resampled['Value'] = df_education_group_resampled['Value'].interpolate(method='linear')  # Interpolating missing values

    # Apply Holt-Winters Exponential Smoothing to the resampled data
    model_hw = ExponentialSmoothing(
        df_education_group_resampled['Value'],
        trend='add',  # You can also try 'mul' for multiplicative trend
        seasonal='add',  # Use 'mul' for multiplicative seasonality if data suggests it
        seasonal_periods=12  # Adjust this based on your data, e.g., if your data shows yearly seasonality
    )

    # Fit the model
    hw_fit = model_hw.fit()

    # Forecast future values
    future_periods = pd.date_range(df_education_group_resampled.index[-1], periods=24, freq='ME')[1:]  # Forecasting the next 12 periods
    forecast_values_hw = hw_fit.forecast(len(future_periods))

    # Set the forecasted future periods as the index for the forecasted values
    forecast_df = pd.DataFrame(forecast_values_hw, index=future_periods, columns=['Forecast'])

    return df_education_group_resampled, hw_fit, forecast_df

# Function to calculate R^2 for combined historical and forecasted data
def calculate_r_squared(df_resampled, hw_fit, forecast_values_hw, future_periods):
    # Combine historical fitted values and forecasted values using pd.concat()
    combined_actuals = pd.concat([df_resampled['Value'], pd.Series([np.nan] * len(future_periods), index=future_periods)])
    combined_fitted_forecasted = pd.concat([hw_fit.fittedvalues, pd.Series(forecast_values_hw, index=future_periods)])

    # Calculate SS_residual (sum of squared residuals) for the entire series
    ss_residual_combined = np.sum((combined_actuals - combined_fitted_forecasted) ** 2)

    # Calculate SS_total (total sum of squares) for the entire series
    ss_total_combined = np.sum((combined_actuals.dropna() - combined_actuals.dropna().mean()) ** 2)

    # Calculate R^2
    r_squared_combined = 1 - (ss_residual_combined / ss_total_combined)

    return r_squared_combined

#Apply Holt-Winters for each education level (Education levels can be found in 'Subgroup')
education_levels = df_filtered['Subgroup'].unique()

education_results = {}

# Loop over each education level and apply the function
for education_level in education_levels:
    df_education_resampled, hw_education_fit, forecast_education_df = apply_holt_winters_for_education_group(education_level)
    education_results[education_level] = {
        'resampled': df_education_resampled,
        'fit': hw_education_fit,
        'forecast': forecast_education_df
    }

    # Calculate R^2 for the education level (combined historical and forecasted data)
    r_squared_education = calculate_r_squared(df_education_resampled, hw_education_fit, forecast_education_df['Forecast'], forecast_education_df.index)

    print(f"R^2 for {education_level}: {r_squared_education:.4f}")

# Step 4: Plot the results for each education level
plt.figure(figsize=(14, 8))
custom_colors = [
    '#1f77b4',  # Blue
    '#ff7f0e',  # Orange
    '#2ca02c',  # Green
    '#d62728',  # Red
    '#9467bd',  # Purple
]

for idx, (education_level, result) in enumerate(education_results.items()):
    color = custom_colors[idx % len(custom_colors)]

    # Plot the observed data (original data points)
    plt.plot(result['resampled'].index, result['resampled']['Value'], label=f'Observed Data ({education_level})',
             color=color)

    # Plot the predicted future values
    plt.plot(result['forecast'].index, result['forecast']['Forecast'], label=f'Predicted Future Values ({education_level})',
             linestyle='--', color=color)

observed_end_date = next(iter(education_results.values()))['resampled'].index[-1]
plt.axvline(x=observed_end_date, color='grey', linewidth = 2.5, linestyle='--', label='Start of Prediction')
plt.axhline(y=11, color='black', linestyle='-', linewidth=2, label='2019 Threshold')
# Adding labels and title
plt.xlabel('Date')
plt.ylabel('Percent of Adults with Symptoms')
plt.title('Comparison of Symptoms of Anxiety Disorder or Depressive Disorder by Education Level Using Holt-Winters Exponential Smoothing')
plt.legend()

# Display the plot
plt.tight_layout()
plt.show()


## FOR RACE

# Load dataset
df = pd.read_csv('anxiety_depression.csv')

# Filter for the Indicator "Symptoms of Anxiety Disorder or Depressive Disorder"
df_filtered = df[df['Indicator'] == 'Symptoms of Anxiety Disorder or Depressive Disorder']

# Filter for the Group "By Race/Hispanic ethnicity"
df_filtered = df_filtered[df_filtered['Group'] == 'By Race/Hispanic ethnicity']

# Define a function to apply Holt-Winters for a specific race subgroup
def apply_holt_winters_for_race_group(race_group):
    # Filter by race/ethnicity subgroup
    df_race_group = df_filtered[df_filtered['Subgroup'] == race_group]

    # Filter for Time Periods 1 to 72
    df_race_group = df_race_group[(df_race_group['Time Period'] >= 1) & (df_race_group['Time Period'] <= 72)]


    df_race_group = df_race_group.groupby('Time Period').agg({'Value': 'mean', 'Time Period End Date': 'first'}).reset_index()


    df_race_group['Time Period End Date'] = pd.to_datetime(df_race_group['Time Period End Date'])


    df_race_group.set_index('Time Period End Date', inplace=True)

    #  Resample the time series to a regular frequency (monthly) using interpolation
    df_race_group_resampled = df_race_group.resample('ME').mean()  # Resampling to monthly frequency, using the mean
    df_race_group_resampled['Value'] = df_race_group_resampled['Value'].interpolate(method='linear')  # Interpolating missing values

    # Apply Holt-Winters Exponential Smoothing to the resampled data
    model_hw = ExponentialSmoothing(
        df_race_group_resampled['Value'],
        trend='add',  # You can also try 'mul' for multiplicative trend
        seasonal='add',  # Use 'mul' for multiplicative seasonality if data suggests it
        seasonal_periods=12  # Adjust this based on your data, e.g., if your data shows yearly seasonality
    )

    # Fit the model
    hw_fit = model_hw.fit()


    #  Forecast future values
    future_periods = pd.date_range(df_race_group_resampled.index[-1], periods=24, freq='ME')[1:]  # Forecasting the next 12 periods
    forecast_values_hw = hw_fit.forecast(len(future_periods))

    # Set the forecasted future periods as the index for the forecasted values
    forecast_df = pd.DataFrame(forecast_values_hw, index=future_periods, columns=['Forecast'])

    return df_race_group_resampled, hw_fit, forecast_df


# Function to calculate R^2 for combined historical and forecasted data
def calculate_r_squared(df_resampled, hw_fit, forecast_values_hw, future_periods):
    # Combine historical fitted values and forecasted values using pd.concat()
    combined_actuals = pd.concat(
        [df_resampled['Value'], pd.Series([np.nan] * len(future_periods), index=future_periods)])
    combined_fitted_forecasted = pd.concat([hw_fit.fittedvalues, pd.Series(forecast_values_hw, index=future_periods)])

    # Calculate SS_residual (sum of squared residuals) for the entire series
    ss_residual_combined = np.sum((combined_actuals - combined_fitted_forecasted) ** 2)

    # Calculate SS_total (total sum of squares) for the entire series
    ss_total_combined = np.sum((combined_actuals.dropna() - combined_actuals.dropna().mean()) ** 2)

    # Calculate R^2
    r_squared_combined = 1 - (ss_residual_combined / ss_total_combined)

    return r_squared_combined


# Apply Holt-Winters for each race subgroup (Racial subgroups can be found in 'Subgroup')
race_groups = df_filtered['Subgroup'].unique()


race_results = {}


for race_group in race_groups:
    df_race_resampled, hw_race_fit, forecast_race_df = apply_holt_winters_for_race_group(race_group)
    race_results[race_group] = {
        'resampled': df_race_resampled,
        'fit': hw_race_fit,
        'forecast': forecast_race_df
    }

    # Calculate R^2 for the race group (combined historical and forecasted data)
    r_squared_race = calculate_r_squared(df_race_resampled, hw_race_fit, forecast_race_df['Forecast'],
                                         forecast_race_df.index)

    print(f"R^2 for {race_group}: {r_squared_race:.4f}")


custom_colors = [
    '#1f77b4',  # Blue
    '#ff7f0e',  # Orange
    '#2ca02c',  # Green
    '#d62728',  # Red
    '#9467bd',  # Purple
]

# Plot the results for each race subgroup
plt.figure(figsize=(14, 8))

# Loop over the race groups to plot their results
for idx, (race_group, result) in enumerate(race_results.items()):
    # Select a color from the custom color list (cycling through if more than the list size)
    color = custom_colors[idx % len(custom_colors)]  # Cycling through custom colors if more than available

    # Plot the observed data
    plt.plot(result['resampled'].index, result['resampled']['Value'], label=f'Observed Data ({race_group})',
             color=color)

    # Plot the predicted future values
    plt.plot(result['forecast'].index, result['forecast']['Forecast'], label=f'Predicted Future Values ({race_group})',
             linestyle='--', color=color)
observed_end_date = next(iter(race_results.values()))['resampled'].index[-1]
plt.axvline(x=observed_end_date, color='grey', linewidth = 2.5, linestyle='--', label='Start of Prediction')
plt.axhline(y=11, color='black', linestyle='-', linewidth=2, label='2019 Threshold')
# Adding labels and title
plt.xlabel('Date')
plt.ylabel('Percent of Adults with Symptoms')
plt.title(
    'Comparison of Symptoms of Anxiety Disorder or Depressive Disorder by Race Using Holt-Winters Exponential Smoothing')
plt.legend()

# Display the plot
plt.tight_layout()
plt.show()