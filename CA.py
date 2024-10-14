import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math

# Load Weather Data from CSV
def load_weather_data(csv_file):
    # Assuming the CSV file contains columns: 'wind_speed', 'temperature', 'humidity'
    df = pd.read_csv(csv_file)
    
    # Automatically calculate the grid shape based on the number of rows in the dataset
    num_data_points = df.shape[0]
    grid_size = int(math.sqrt(num_data_points))  # Get the closest square root
    num_cells = grid_size * grid_size  # Total number of cells in the square grid
    
    # Truncate or reshape the data to fit the grid
    truncated_df = df.head(num_cells)
    
    wind_speed = truncated_df['Wind Speed (km/h)'].values.reshape(grid_size, grid_size)
    temperature = truncated_df['Temperature (C)'].values.reshape(grid_size, grid_size)
    humidity = truncated_df['Humidity'].values.reshape(grid_size, grid_size)
    
    weather_data = {
        "wind_speed": wind_speed,
        "temperature": temperature,
        "humidity": humidity,
        "grid_size": grid_size
    }
    
    return weather_data

# Define grid structure and neighborhood (Moore Neighborhood)
def get_neighbors(i, j, grid, grid_size):
    neighbors = []
    for x in range(-1, 2):
        for y in range(-1, 2):
            if x == 0 and y == 0:
                continue  # Skip the center cell itself
            ni, nj = i + x, j + y
            if 0 <= ni < grid_size and 0 <= nj < grid_size:
                neighbors.append(grid[ni, nj])
    return neighbors

# Stochastic Initial State Distribution (0: Sunny, 1: Stormy)
def initialize_grid(grid_size, p_stormy=0.2):
    grid = np.zeros((grid_size, grid_size), dtype=int)
    for i in range(grid_size):
        for j in range(grid_size):
            grid[i, j] = 1 if random.random() < p_stormy else 0  # Stormy with probability p_stormy
    return grid

# Transition Rules
def transition(cell_state, neighbors, wind_speed, temperature, humidity):
    stormy_neighbors = sum(neighbors)
    random_chance = random.random()

    if cell_state == 1:  # Stormy
        if wind_speed > 50 and temperature < 10 and humidity > 70 and random_chance < 0.7:
            return 1  # Stay Stormy
        elif wind_speed > 30 and stormy_neighbors >= 2 and random_chance < 0.5:
            return 0  # Transition to Sunny
        elif wind_speed < 30 and sum(neighbors) < len(neighbors) / 2 and random_chance < 0.4:
            return 0  # Transition to Sunny
    elif cell_state == 0:  # Sunny
        if wind_speed < 30 and temperature > 20 and humidity < 50 and random_chance < 0.8:
            return 0  # Stay Sunny
        elif wind_speed > 50 and stormy_neighbors >= 2 and random_chance < 0.5:
            return 1  # Transition to Stormy

    return cell_state  # No change

# Simulation Execution
def run_simulation(grid, weather_data, iterations=10):
    grids = [grid.copy()]  # Store grids for visualization
    grid_size = weather_data['grid_size']
    
    for _ in range(iterations):
        new_grid = grid.copy()
        for i in range(grid_size):
            for j in range(grid_size):
                neighbors = get_neighbors(i, j, grid, grid_size)
                wind_speed = weather_data['wind_speed'][i, j]
                temperature = weather_data['temperature'][i, j]
                humidity = weather_data['humidity'][i, j]

                # Apply transition rules
                new_grid[i, j] = transition(grid[i, j], neighbors, wind_speed, temperature, humidity)

        grid = new_grid.copy()  # Update grid for next iteration
        grids.append(grid.copy())  # Save grid state for each iteration
    return grids

# Validation and Analysis
def analyze_grid(grid):
    unique, counts = np.unique(grid, return_counts=True)
    distribution = dict(zip(unique, counts))
    print(f"Final State Distribution: {distribution}")
    return distribution

# Visualize the grid using matplotlib
def visualize_actual_vs_prediction(actual_grid, predicted_grids):
    cmap = mcolors.ListedColormap (['yellow', 'blue'])  # Yellow for Sunny, Blue for Stormy
    bounds = [-0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, axes = plt.subplots(1, len(predicted_grids) + 1, figsize=(15, 3))

    # First visual: actual weather data
    axes[0].imshow(actual_grid, cmap=cmap, norm=norm)
    axes[0].set_title("Actual Weather")
    axes[0].axis('off')

    # Next visual: predictions
    for i in range(len(predicted_grids)):
        axes[i + 1].imshow(predicted_grids[i], cmap=cmap, norm=norm)
        axes[i + 1].set_title("Prediction")
        axes[i + 1].axis('off')

    plt.tight_layout()
    plt.show()


# Main Program
if __name__ == "__main__":
    # Load weather data from CSV file
    csv_file = 'weatherHistory.csv'  # Replace with your CSV file path
    weather_data = load_weather_data(csv_file)

    # Initialize the grid with stochastic distribution
    grid_size = weather_data['grid_size']
    grid = initialize_grid(grid_size)

    # Run the simulation for 1 iteration
    grids = run_simulation(grid, weather_data, iterations=1)

    # Analyze the final grid
    analyze_grid(grids[-1])

    # Visualize the grid states over iteration
    visualize_actual_vs_prediction(grid, grids[1:])
