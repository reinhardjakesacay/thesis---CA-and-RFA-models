import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Read the CSV file and prepare data
df = pd.read_csv('weatherHistory.csv', encoding='cp1252')

# Feature extraction for training Random Forest model
df['Wind Level'] = np.where(df['Wind Speed (km/h)'] <= df['Wind Speed (km/h)'].median(), 'Low', 'High')
df['Temp Level'] = np.where(df['Temperature'] <= df['Temperature'].median(), 'Low', 'High')
df['Humidity Level'] = np.where(df['Humidity'] >= df['Humidity'].median(), 'High', 'Low')

# Map features to numeric values
wind_map = {'Low': 0, 'High': 1}
temp_map = {'Low': 0, 'High': 1}
humidity_map = {'Low': 0, 'High': 1}
df['Wind Level'] = df['Wind Level'].map(wind_map)
df['Temp Level'] = df['Temp Level'].map(temp_map)
df['Humidity Level'] = df['Humidity Level'].map(humidity_map)

# Create training dataset (assuming random labels for demonstration purposes)
X = df[['Wind Level', 'Temp Level', 'Humidity Level']]
y = np.random.randint(0, 2, size=len(X))  # Random binary outcome (0 = no storm, 1 = storm)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# CA Simulation Parameters
grid_size = 200
steps = 101
weather_grid = np.zeros((grid_size, grid_size))
storm_x, storm_y = np.random.randint(0, grid_size), int(grid_size * 0.75)
weather_grid[storm_x, storm_y] = 1

# Initialize grids
wind_grid = np.random.choice(['Low', 'High'], size=(grid_size, grid_size))
temp_grid = np.random.choice(['Low', 'High'], size=(grid_size, grid_size))
humidity_grid = np.random.choice(['Low', 'High'], size=(grid_size, grid_size))

def is_within_circular_spread(x, y, storm_x, storm_y, radius):
    return (x - storm_x) ** 2 + (y - storm_y) ** 2 <= radius ** 2

def update_weather_with_model(grid, wind_grid, temp_grid, humidity_grid, storm_x, storm_y, radius, model):
    new_grid = grid.copy()
    for i in range(grid_size):
        for j in range(grid_size):
            if is_within_circular_spread(i, j, storm_x, storm_y, radius):
                wind = wind_map[wind_grid[i, j]]
                temp = temp_map[temp_grid[i, j]]
                humidity = humidity_map[humidity_grid[i, j]]
                
                # Predict storm probability
                features_df = pd.DataFrame([[wind, temp, humidity]], columns=['Wind Level', 'Temp Level', 'Humidity Level'])
                prob_storm = model.predict_proba(features_df)[0, 1]

                
                # Update grid based on prediction
                if prob_storm > 0.5:  # Threshold can be adjusted
                    new_grid[i, j] = 1
                else:
                    new_grid[i, j] = 0

    # Update storm center position randomly
    storm_y = (storm_y - 1) % grid_size
    storm_x = (storm_x + np.random.randint(-3, 4)) % grid_size
    new_grid[storm_x, storm_y] = 1  
    
    return new_grid, storm_x, storm_y

# Set up for visualization
cmap = colors.ListedColormap(['blue', 'gray'])
norm = colors.BoundaryNorm([0, 0.5, 1], cmap.N)
fig, ax = plt.subplots(figsize=(6, 6))
storm_radius = 1

# Run the simulation
for step in range(steps):
    weather_grid, storm_x, storm_y = update_weather_with_model(
        weather_grid, wind_grid, temp_grid, humidity_grid, storm_x, storm_y, storm_radius, model
    )
    storm_radius += 0.2
    ax.clear()
    ax.imshow(weather_grid, cmap=cmap, norm=norm)
    ax.axis('off')
    plt.pause(0.1)

# Save the final picture
plt.savefig('Hybrid_Model_RFA.png', bbox_inches='tight', pad_inches=0)
plt.show()
