import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

# Read the CSV file and get min, max, and random values
df = pd.read_csv('weatherHistory.csv', encoding='cp1252')
random_temperature = df['Temperature'].sample().values[0]
random_wind_speed = df['Wind Speed (km/h)'].sample().values[0]
random_humidity = df['Humidity'].sample().values[0]

max_temperature = df['Temperature'].max()
min_temperature = df['Temperature'].min()
max_wind_speed = df['Wind Speed (km/h)'].max()
min_wind_speed = df['Wind Speed (km/h)'].min()
max_humidity = df['Humidity'].max()
min_humidity = df['Humidity'].min()

# CA Simulation Parameters
grid_size = 200
steps = 101
wind_speeds = ['Low', 'Medium', 'High']  
temperatures = ['Low', 'High']  
humidity_levels = ['Low', 'High']  

# Create initial grid and weather conditions
weather_grid = np.zeros((grid_size, grid_size))
storm_x = np.random.randint(0, grid_size)
storm_y = int(grid_size * 0.75)
weather_grid[storm_x, storm_y] = 1  

# Set up probability based on CSV values
wind_prob = [0.08, 0.7, 0.22] #if random_wind_speed <= max_wind_speed/2 else [0.2, 0.5, 0.3]
temp_prob = [0.2, 0.8] #if random_temperature <= max_temperature/2 else [0.3, 0.7]
humidity_prob = [0.9, 0.1]  #if random_humidity >= min_humidity else [0.3, 0.7]

wind_grid = np.random.choice(wind_speeds, size=(grid_size, grid_size), p=wind_prob)
temp_grid = np.random.choice(temperatures, size=(grid_size, grid_size), p=temp_prob)
humidity_grid = np.random.choice(humidity_levels, size=(grid_size, grid_size), p=humidity_prob)

def is_within_circular_spread(x, y, storm_x, storm_y, radius):
    return (x - storm_x)**2 + (y - storm_y)**2 <= radius**2

def update_weather(grid, wind_grid, temp_grid, humidity_grid, storm_x, storm_y, radius):
    new_grid = grid.copy()

    for i in range(grid_size):
        for j in range(grid_size):
            state = grid[i, j]
            wind = wind_grid[i, j]
            temp = temp_grid[i, j]
            humidity = humidity_grid[i, j]
            random_chance = np.random.uniform(0, 1)
            
            if state == 1:
                if wind == 'High' and temp == 'Low' and humidity == 'High' and random_chance < 0.7:
                    new_grid[i, j] = 1
                elif wind == 'Medium' and random_chance < 0.5:
                    new_grid[i, j] = 0
                elif wind == 'Low' and random_chance < 0.4:
                    new_grid[i, j] = 0

            elif state == 0:
                if is_within_circular_spread(i, j, storm_x, storm_y, radius):
                    if wind == 'High' and random_chance < 0.5:
                        new_grid[i, j] = 1
                    elif wind == 'Medium' and random_chance < 0.4:
                        new_grid[i, j] = 1

    storm_y = (storm_y - 1) % grid_size
    storm_x = (storm_x + np.random.randint(-3, 4)) % grid_size
    new_grid[storm_x, storm_y] = 1  
    
    return new_grid, storm_x, storm_y

cmap = colors.ListedColormap(['blue', 'gray'])
norm = colors.BoundaryNorm([0, 0.5, 1], cmap.N)

fig, ax = plt.subplots(figsize=(6, 6))
storm_radius = 1

for step in range(steps):
    weather_grid, storm_x, storm_y = update_weather(weather_grid, wind_grid, temp_grid, humidity_grid, storm_x, storm_y, storm_radius)
    storm_radius += 0.2
    ax.clear()
    ax.imshow(weather_grid, cmap=cmap, norm=norm)
    
    # Remove x and y axis labels
    ax.axis('off')

    plt.pause(0.1)

# Save the final picture
plt.savefig('Reg_CA_Model.png', bbox_inches='tight', pad_inches=0)
plt.show()
