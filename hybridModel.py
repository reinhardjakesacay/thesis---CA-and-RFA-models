import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import pandas as pd
import os

# Read the weather data from the CSV file
df = pd.read_csv('weatherHistory.csv', encoding='cp1252')

# Extract relevant columns and normalize the data if needed
temperature_data = df['Temperature (C)'].to_numpy()  # Temperature in °C
humidity_data = df['Humidity'].to_numpy()  # Humidity (assumed to be between 0 and 1)
wind_speed_data = df['Wind Speed (km/h)'].to_numpy()  # Wind speed in km/h

# Parameters
grid_size = 100
steps = 100
wind_influence = 0.2
temp_diffusion = 0.1
humidity_initial = 0.5  # Initial humidity level

# Initialize grids using the data
grid = np.resize(temperature_data, (grid_size, grid_size))
humidity = np.resize(humidity_data, (grid_size, grid_size))
wind_u = np.resize(np.random.uniform(low=-1, high=1, size=(grid_size * grid_size)), (grid_size, grid_size))  # Wind in x-direction
wind_v = np.resize(np.random.uniform(low=-1, high=1, size=(grid_size * grid_size)), (grid_size, grid_size))  # Wind in y-direction

# Initialize cloud cover (1 = cloud, 0 = no cloud)
clouds = np.zeros((grid_size, grid_size))
clouds[0:5, :] = 1  # Simulating cloud cover starting from the left

# Typhoon initialization
typhoon_x, typhoon_y = grid_size // 2, grid_size // 2  # Initial position of the typhoon
typhoon_strength = 2  # Strength of the typhoon's wind effect

# List to collect data for each step
collected_data = []

# Define the update rules for each step
def update_grid(grid, wind_u, wind_v, clouds, humidity, typhoon_x, typhoon_y):
    new_grid = grid.copy()
    new_clouds = np.zeros_like(clouds)
    
    # Typhoon movement (now moving upwards - reverse direction)
    typhoon_y = (typhoon_y - 1) % grid_size  # Move typhoon upwards
    typhoon_x = (typhoon_x + np.random.randint(-1, 2)) % grid_size  # Random horizontal movement
    
    # Apply typhoon influence on wind with stochasticity
    for i in range(max(typhoon_x - 3, 0), min(typhoon_x + 3, grid_size)):
        for j in range(max(typhoon_y - 3, 0), min(typhoon_y + 3, grid_size)):
            # Introduce stochasticity in wind influence
            wind_u[i, j] += np.random.uniform(-typhoon_strength, typhoon_strength)
            wind_v[i, j] += np.random.uniform(-typhoon_strength, typhoon_strength)
    
    # Update temperature and humidity based on diffusion and wind
    for i in range(1, grid_size - 1):
        for j in range(1, grid_size - 1):
            temp_avg = (grid[i - 1, j] + grid[i + 1, j] + grid[i, j - 1] + grid[i, j + 1]) / 4
            wind_effect = wind_influence * (wind_u[i, j] + wind_v[i, j])
            # Introduce stochasticity in temperature diffusion
            random_factor = np.random.uniform(-0.1, 0.1)  # Random fluctuation
            new_grid[i, j] += temp_diffusion * (temp_avg - grid[i, j]) + wind_effect + random_factor
            
            # Update humidity based on temperature and clouds with stochasticity
            humidity_change = 0.1 if clouds[i, j] == 1 else -0.01
            humidity_change += np.random.uniform(-0.05, 0.05)  # Add randomness to humidity change
            new_humidity = humidity[i, j] + humidity_change
            new_humidity = np.clip(new_humidity, 0, 1)  # Keep humidity within [0, 1]
            humidity[i, j] = new_humidity
            
            # Update cloud movement
            new_i = (i + int(wind_v[i, j])) % grid_size
            new_j = (j + int(wind_u[i, j])) % grid_size
            new_clouds[new_i, new_j] = clouds[i, j]

    return new_grid, new_clouds, humidity, typhoon_x, typhoon_y

# Create color map for visualization
cmap_temp = colors.ListedColormap(['blue', 'green', 'yellow', 'orange', 'red'])
bounds_temp = [0, 15, 20, 25, 30, 35]
norm_temp = colors.BoundaryNorm(bounds_temp, cmap_temp.N)

cmap_cloud = colors.ListedColormap(['white', 'gray'])  # White for no clouds, gray for clouds
norm_cloud = colors.BoundaryNorm([0, 0.5, 1], cmap_cloud.N)

# Simulation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

for step in range(steps):
    grid, clouds, humidity, typhoon_x, typhoon_y = update_grid(grid, wind_u, wind_v, clouds, humidity, typhoon_x, typhoon_y)
    
    # Collect data for the current step (mean temperature, cloud coverage, wind speed, humidity, and storm percentage)
    mean_temperature = np.mean(grid)
    cloud_coverage = np.mean(clouds)
    wind_speed = np.sqrt(wind_u**2 + wind_v**2)  # Calculate wind speed as the magnitude of wind vectors
    mean_wind_speed = np.mean(wind_speed)
    storm_percentage = np.sum(clouds) / (grid_size ** 2) * 100  # Calculate storm percentage

    collected_data.append([step, mean_temperature, cloud_coverage, mean_wind_speed, np.mean(humidity), storm_percentage])
    
    # Plot temperature
    ax1.clear()
    ax1.imshow(grid, cmap=cmap_temp, norm=norm_temp)
    ax1.set_title(f"Temperature (Step {step + 1})")
    
    # Plot clouds
    ax2.clear()
    ax2.imshow(clouds, cmap=cmap_cloud, norm=norm_cloud)
    ax2.set_title(f"Clouds and Typhoon (Step {step + 1})")
    
    # Mark typhoon on cloud map
    ax2.scatter(typhoon_y, typhoon_x, color='red', s=100, label='Typhoon')
    ax2.legend()
    plt.pause(0.1)

# Check for existing file and create a new one if necessary
base_filename = 'CA_RFA_output.csv'
new_filename = base_filename
counter = 1

while os.path.exists(new_filename):
    new_filename = f'CA_RFA_output_{counter}.csv'
    counter += 1

# After the simulation, save collected data to CSV
output_data = pd.DataFrame(collected_data, columns=['Step', 'Mean Temperature (°C)', 'Cloud Coverage', 'Mean Wind Speed', 'Mean Humidity', 'Storm Percentage (%)'])
output_data.to_csv(new_filename, index=False)

plt.show()
