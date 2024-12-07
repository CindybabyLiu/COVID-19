import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

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

#print(case_data)

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

