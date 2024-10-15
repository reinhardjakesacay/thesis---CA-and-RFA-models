import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

# Parameters
grid_size = 100
steps = 100
wind_influence = 0.2
temp_diffusion = 0.1

# Initialize grid (temperature in °C)
grid = np.random.uniform(low=15, high=30, size=(grid_size, grid_size))

# Wind velocity grid (representing directional wind strength)
wind_u = np.random.uniform(low=-1, high=1, size=(grid_size, grid_size))  # Wind in x-direction
wind_v = np.random.uniform(low=-1, high=1, size=(grid_size, grid_size))  # Wind in y-direction

# Initialize cloud cover (1 = cloud, 0 = no cloud)
clouds = np.zeros((grid_size, grid_size))
clouds[0:5, :] = 1  # Simulating cloud cover starting from the left

# Typhoon initialization
typhoon_x, typhoon_y = grid_size // 2, grid_size // 2  # Initial position of the typhoon
typhoon_strength = 2  # Strength of the typhoon's wind effect

# Define the update rules for each step
def update_grid(grid, wind_u, wind_v, clouds, typhoon_x, typhoon_y):
    new_grid = grid.copy()
    new_clouds = np.zeros_like(clouds)
    
    # Typhoon movement (now moving upwards - reverse direction)
    typhoon_y = (typhoon_y - 1) % grid_size  # Move typhoon upwards
    typhoon_x = (typhoon_x + np.random.randint(-1, 2)) % grid_size  # Random horizontal movement
    
    # Apply typhoon influence on wind
    for i in range(max(typhoon_x-3, 0), min(typhoon_x+3, grid_size)):
        for j in range(max(typhoon_y-3, 0), min(typhoon_y+3, grid_size)):
            wind_u[i, j] += np.random.uniform(-typhoon_strength, typhoon_strength)
            wind_v[i, j] += np.random.uniform(-typhoon_strength, typhoon_strength)
    
    # Update temperature based on diffusion and wind
    for i in range(1, grid_size-1):
        for j in range(1, grid_size-1):
            temp_avg = (grid[i-1, j] + grid[i+1, j] + grid[i, j-1] + grid[i, j+1]) / 4
            wind_effect = wind_influence * (wind_u[i, j] + wind_v[i, j])
            new_grid[i, j] += temp_diffusion * (temp_avg - grid[i, j]) + wind_effect
            
            # Update cloud movement
            new_i = (i + int(wind_v[i, j])) % grid_size
            new_j = (j + int(wind_u[i, j])) % grid_size
            new_clouds[new_i, new_j] = clouds[i, j]

    return new_grid, new_clouds, typhoon_x, typhoon_y

# Create color map for visualization
cmap_temp = colors.ListedColormap(['blue', 'green', 'yellow', 'orange', 'red'])
bounds_temp = [0, 15, 20, 25, 30, 35]
norm_temp = colors.BoundaryNorm(bounds_temp, cmap_temp.N)

cmap_cloud = colors.ListedColormap(['white', 'gray'])  # White for no clouds, gray for clouds
norm_cloud = colors.BoundaryNorm([0, 0.5, 1], cmap_cloud.N)

# Simulation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

for step in range(steps):
    grid, clouds, typhoon_x, typhoon_y = update_grid(grid, wind_u, wind_v, clouds, typhoon_x, typhoon_y)
    
    # Plot temperature
    ax1.clear()
    ax1.imshow(grid, cmap=cmap_temp, norm=norm_temp)
    ax1.set_title(f"Temperature (Step {step})")
    
    # Plot clouds
    ax2.clear()
    ax2.imshow(clouds, cmap=cmap_cloud, norm=norm_cloud)
    ax2.set_title(f"Clouds and Typhoon (Step {step})")
    
    # Mark typhoon on cloud map
    ax2.scatter(typhoon_y, typhoon_x, color='red', s=100, label='Typhoon')
    plt.pause(0.1)

plt.show()
