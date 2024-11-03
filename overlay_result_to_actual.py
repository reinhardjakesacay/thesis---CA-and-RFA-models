import cv2
import matplotlib.pyplot as plt
import numpy as np

# Ensure that the results are cropped before using for overlay
# Read the images
ca_img = cv2.imread('cropped_CA_for_overlay.png')
rfa_img = cv2.imread('cropped_CA-RFA_for_overlay.png')
actual_img = cv2.imread('cropped_actual_stormtrack_for_overlay.png')

# Convert images to RGB for Matplotlib
ca_img = cv2.cvtColor(ca_img, cv2.COLOR_BGR2RGB)
rfa_img = cv2.cvtColor(rfa_img, cv2.COLOR_BGR2RGB)
actual_img = cv2.cvtColor(actual_img, cv2.COLOR_BGR2RGB)

# Ensure all images have the same size
height, width = actual_img.shape[:2]
ca_img = cv2.resize(ca_img, (width, height))
rfa_img = cv2.resize(rfa_img, (width, height))

# Create a blank image to overlay the predictions with the same size as the actual image
overlay_img = np.zeros_like(actual_img)

# Overlay the predictions with different colors
overlay_img[np.where((ca_img != 0).all(axis=2))] = [255, 50, 50]  # Bright red for CA
overlay_img[np.where((rfa_img != 0).all(axis=2))] = [255, 255, 50]  # Bright yellow for CA-RFA
overlay_img[np.where((actual_img != 0).all(axis=2))] = [50, 255, 50]  # Bright green for actual

# Combine the overlay with the actual storm track
combined_img = cv2.addWeighted(actual_img, 0.4, overlay_img, 0.6, 0)

# Display the combined image
plt.figure(figsize=(10, 10))
plt.imshow(combined_img)
plt.axis('off')
plt.title('Enhanced Comparison of CA, CA-RFA, and Actual Storm Track')
plt.show()
