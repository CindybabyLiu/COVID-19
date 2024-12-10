import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error

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

