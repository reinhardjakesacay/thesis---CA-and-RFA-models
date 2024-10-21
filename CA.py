import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import pandas as pd

# Read the CSV file
df = pd.read_csv('weatherHistory.csv', encoding='cp1252')

# Print the first 5 rows of 'Column1' and 'Column2'
print(df[['Temperature', 'Humidity', 'Wind Speed (km/h)']].head(5))


# Parameters
grid_size = 200
steps = 100
wind_speeds = ['Low', 'Medium', 'High']  # Wind speeds
temperatures = ['Low', 'High']  # Temperature categories
humidity_levels = ['Low', 'High']  # Humidity levels

# Initialize grid (0 = Sunny, 1 = Stormy)
weather_grid = np.zeros((grid_size, grid_size))

# Introduce storm starting as a single point, slightly to the left on the right side
storm_x = grid_size // 2  # Vertical position of the storm (middle row)
storm_y = int(grid_size * 0.75)  # Horizontal position (75% of the grid width, slightly left from the far right)
weather_grid[storm_x, storm_y] = 1  # Initial storm point

# Wind, Temperature, and Humidity grids
wind_grid = np.random.choice(wind_speeds, size=(grid_size, grid_size), p=[0.4, 0.4, 0.2])
temp_grid = np.random.choice(temperatures, size=(grid_size, grid_size), p=[0.5, 0.5])
humidity_grid = np.random.choice(humidity_levels, size=(grid_size, grid_size), p=[0.5, 0.5])

# Function to count stormy neighbors and check distance from storm center for circular spread
def is_within_circular_spread(x, y, storm_x, storm_y, radius):
    return (x - storm_x)**2 + (y - storm_y)**2 <= radius**2

# Define the update rules based on storm conditions
def update_weather(grid, wind_grid, temp_grid, humidity_grid, storm_x, storm_y, radius):
    new_grid = grid.copy()

    for i in range(grid_size):
        for j in range(grid_size):
            state = grid[i, j]
            wind = wind_grid[i, j]
            temp = temp_grid[i, j]
            humidity = humidity_grid[i, j]
            random_chance = np.random.uniform(0, 1)
            
            # Rules for Stormy (ST) to Stormy (ST)
            if state == 1:  # Currently stormy
                if wind == 'High' and temp == 'Low' and humidity == 'High' and random_chance < 0.7:
                    new_grid[i, j] = 1  # Stay stormy
                # Stormy (ST) to Sunny (SU)
                elif wind == 'Medium' and random_chance < 0.5:
                    new_grid[i, j] = 0  # Change to Sunny
                elif wind == 'Low' and random_chance < 0.4:
                    new_grid[i, j] = 0  # Change to Sunny

            # Rules for Sunny (SU) to Stormy (ST)
            elif state == 0:  # Currently sunny
                if is_within_circular_spread(i, j, storm_x, storm_y, radius):
                    if wind == 'High' and random_chance < 0.5:
                        new_grid[i, j] = 1  # Change to Stormy
                    elif wind == 'Medium' and random_chance < 0.4:
                        new_grid[i, j] = 1  # Change to Stormy

    # Move the storm leftward and unpredictably up/down
    storm_y = (storm_y - 1) % grid_size  # Move left
    storm_x = (storm_x + np.random.randint(-3, 4)) % grid_size  # Random vertical movement
    
    new_grid[storm_x, storm_y] = 1  # Ensure storm keeps moving unpredictably
    
    return new_grid, storm_x, storm_y

# Create color map for visualization (Blue for sunny, Gray for stormy)
cmap = colors.ListedColormap(['blue', 'gray'])
norm = colors.BoundaryNorm([0, 0.5, 1], cmap.N)

# Simulation
fig, ax = plt.subplots(figsize=(6, 6))

# Initial radius of the storm
storm_radius = 1

for step in range(steps):
    weather_grid, storm_x, storm_y = update_weather(weather_grid, wind_grid, temp_grid, humidity_grid, storm_x, storm_y, storm_radius)
    
    # Increase storm radius as it moves to simulate spread
    storm_radius += 0.2
    
    # Plot weather states
    ax.clear()
    ax.imshow(weather_grid, cmap=cmap, norm=norm)
    ax.set_title(f"Storm Prediction (Step {step})")
    plt.pause(0.1)

plt.show()
