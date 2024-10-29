import pandas as pd

# Load the CSV file
df = pd.read_csv('weatherHistory.csv', encoding='cp1252')

# Sample random values
random_temperature = df['Temperature'].sample().values[0]
random_wind_speed = df['Wind Speed (km/h)'].sample().values[0]
random_humidity = df['Humidity'].sample().values[0]

# Calculate min, max for temperature, humidity, and wind speed
max_temperature = df['Temperature'].max()
min_temperature = df['Temperature'].min()
max_humidity = df['Humidity'].max()
min_humidity = df['Humidity'].min()
max_wind_speed = df['Wind Speed (km/h)'].max()
min_wind_speed = df['Wind Speed (km/h)'].min()

# Define thresholds
temp_step = (max_temperature - min_temperature) / 3
low_temp_threshold = min_temperature + temp_step
high_temp_threshold = min_temperature + 2 * temp_step
mid_wind_speed = (max_wind_speed + min_wind_speed) / 2
mid_hum = (max_humidity + min_humidity) / 2

# Classification functions
def classify_temperature(temp):
    if temp < low_temp_threshold:
        return 'Low'
    elif temp < high_temp_threshold:
        return 'Medium'
    else:
        return 'High'

def classify_humidity(humidity):
    return 'Low' if humidity < mid_hum else 'High'

def classify_wind_speed(speed):
    return 'Low' if speed < mid_wind_speed else 'High'

# Apply classifications to each row
df['Temperature_Class'] = df['Temperature'].apply(classify_temperature)
df['Humidity_Class'] = df['Humidity'].apply(classify_humidity)
df['Wind_Speed_Class'] = df['Wind Speed (km/h)'].apply(classify_wind_speed)

# Calculate the percentage of each classification
total_count = len(df)
temp_percentages = (df['Temperature_Class'].value_counts() / total_count) * 100
humidity_percentages = (df['Humidity_Class'].value_counts() / total_count) * 100
wind_speed_percentages = (df['Wind_Speed_Class'].value_counts() / total_count) * 100

# Count values in each category
temp_counts = df['Temperature_Class'].value_counts()
humidity_counts = df['Humidity_Class'].value_counts()
wind_speed_counts = df['Wind_Speed_Class'].value_counts()

# Print results
print("Temperature Classification Counts:")
print(temp_counts)
print("Temperature Classification Percentages:")
print(temp_percentages)


print("\nHumidity Classification Counts:")
print(humidity_counts)
print("\nHumidity Classification Percentages:")
print(humidity_percentages)


print("\nWind Speed Classification Counts:")
print(wind_speed_counts)
print("\nWind Speed Classification Percentages:")
print(wind_speed_percentages)
