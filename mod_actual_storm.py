import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the original image
image_path = 'storm_track.png'
original_img = cv2.imread(image_path)

# Convert to HSV color space for color-based segmentation
hsv_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)

# Define color range to isolate the storm track (adjust these values if needed)
lower_color = np.array([0, 0, 120])    # Lower HSV bound for storm path color
upper_color = np.array([180, 50, 255]) # Upper HSV bound for storm path color

# Create a mask to isolate the storm track
mask = cv2.inRange(hsv_img, lower_color, upper_color)

# Remove small noise by applying morphological operations
kernel = np.ones((5, 5), np.uint8)  # Increased kernel size for stronger noise removal
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Close small holes inside the storm track
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Remove small objects from the mask

# Resize to a 200x200 grid
resized_mask = cv2.resize(mask, (200, 200), interpolation=cv2.INTER_NEAREST)

# Create a blue background for the entire 200x200 grid
blue_background = np.full((200, 200, 3), (0, 0, 255), dtype=np.uint8)  # RGB for blue background

# Overlay the gray storm track onto the blue background
# Set the storm track areas (where mask is 255) to gray
result_img = np.where(resized_mask[..., None] == 255, (128, 128, 128), blue_background)

# Plot the final result
plt.figure(figsize=(6, 6))
plt.imshow(result_img)
plt.title('Processed Actual Storm Track')
plt.axis('off')
plt.show()
