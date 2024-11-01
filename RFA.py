import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Function to check if a point is within a circular spread
def is_within_circular_spread(x, y, storm_x, storm_y, radius):
    return (x - storm_x)**2 + (y - storm_y)**2 <= radius**2

# Function to update the weather grid based on CA rules
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

# Initialize parameters
grid_size = 200
steps = 101
wind_speeds = ['Low', 'Medium', 'High']
temperatures = ['Low', 'High']
humidity_levels = ['Low', 'High']

# Create initial grid and weather conditions
weather_grid = np.zeros((grid_size, grid_size))
storm_x = grid_size // 2
storm_y = int(grid_size * 0.75)
weather_grid[storm_x, storm_y] = 1  

# Initialize wind, temperature, and humidity grids
wind_grid = np.random.choice(wind_speeds, size=(grid_size, grid_size), p=[0.08, 0.7, 0.22])
temp_grid = np.random.choice(temperatures, size=(grid_size, grid_size), p=[0.2, 0.8])
humidity_grid = np.random.choice(humidity_levels, size=(grid_size, grid_size), p=[0.9, 0.1])

# Data collection
data_records = []
storm_radius = 1

for step in range(steps):
    weather_grid, storm_x, storm_y = update_weather(weather_grid, wind_grid, temp_grid, humidity_grid, storm_x, storm_y, storm_radius)
    
    # Record features and storm position
    features = {
        'step': step,
        'storm_x': storm_x,
        'storm_y': storm_y,
        'storm_radius': storm_radius,
        'wind_speed': np.mean(np.where(wind_grid == 'High', 1, np.where(wind_grid == 'Medium', 0.5, 0))),  # Convert wind to numerical
        'temperature': np.mean(np.where(temp_grid == 'High', 1, 0)),  # Convert temp to numerical
        'humidity': np.mean(np.where(humidity_grid == 'High', 1, 0)),  # Convert humidity to numerical
    }
    data_records.append(features)
    
    storm_radius += 0.2  # Increment storm radius

# Convert records to DataFrame
data_df = pd.DataFrame(data_records)

# Prepare the dataset for Random Forest Model
X = data_df[['step', 'storm_radius', 'wind_speed', 'temperature', 'humidity']]
y = data_df[['storm_x', 'storm_y']]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make Predictions
y_pred = rf_model.predict(X_test)

# Calculate and print the Mean Squared Error of the predictions
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Plot the predicted positions vs actual positions
plt.figure(figsize=(10, 6))
plt.scatter(y_test['storm_x'], y_test['storm_y'], color='blue', label='Actual Position')
plt.scatter(y_pred[:, 0], y_pred[:, 1], color='red', label='Predicted Position')
plt.xlabel('Storm X Position')
plt.ylabel('Storm Y Position')
plt.title('Actual vs Predicted Storm Positions')
plt.legend()
plt.grid()
plt.show()

# Visualization of the storm progression
cmap = colors.ListedColormap(['blue', 'gray'])
norm = colors.BoundaryNorm([0, 0.5, 1], cmap.N)

# Initialize the figure for storm visualization
fig, ax = plt.subplots(figsize=(6, 6))

for step in range(steps):
    weather_grid, storm_x, storm_y = update_weather(weather_grid, wind_grid, temp_grid, humidity_grid, storm_x, storm_y, storm_radius)
    
    # Clear the previous plot and update the image
    ax.clear()
    ax.imshow(weather_grid, cmap=cmap, norm=norm)
    ax.set_title(f"Storm Prediction (Step {step})")
    
    # Remove x and y axis labels
    ax.axis('off')

    plt.pause(0.1)

# Save the final picture
plt.savefig('storm_prediction_final.png', bbox_inches='tight', pad_inches=0)
plt.show()
