import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
image_path = r"C:\onedive backup\Desktop\New folder\train\mange\mange-12-_jpg.rf.1ad5e4ea9eaf283df3cd94cc9651a966.jpg"
image = cv2.imread(image_path)

# Convert to HSV (Hue, Saturation, Value) color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Get the middle pixel of the image
height, width, _ = image.shape
middle_x = width // 2
middle_y = height // 2

# Get the HSV value of the middle pixel
middle_hsv = hsv[middle_y, middle_x]

# Define the color range for affected skin based on the middle pixel
lower_affected_skin = np.array([middle_hsv[0] - 10, middle_hsv[1] - 50, middle_hsv[2] - 50])
upper_affected_skin = np.array([middle_hsv[0] + 10, middle_hsv[1] + 50, middle_hsv[2] + 50])

# Create a mask for affected skin
mask_affected_skin = cv2.inRange(hsv, lower_affected_skin, upper_affected_skin)

# Apply morphological operations to refine the mask
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
mask_affected_skin = cv2.erode(mask_affected_skin, kernel, iterations=2)
mask_affected_skin = cv2.dilate(mask_affected_skin, kernel, iterations=2)

# Find contours of the affected areas
contours, _ = cv2.findContours(mask_affected_skin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a mask for the contours
contour_mask = np.zeros_like(image)
cv2.drawContours(contour_mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

# Extract the regions of interest from the original image using the mask
extracted_image = cv2.bitwise_and(image, contour_mask)

# Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance the contrast
lab = cv2.cvtColor(extracted_image, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
cl = clahe.apply(l)
limg = cv2.merge((cl,a,b))
enhanced_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

# Apply unsharp masking to sharpen the image
gaussian_blur = cv2.GaussianBlur(enhanced_image, (5, 5), 0)
unsharp_mask = cv2.addWeighted(enhanced_image, 1.5, gaussian_blur, -0.5, 0)

# Display the result
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(unsharp_mask, cv2.COLOR_BGR2RGB))
plt.title('Enhanced Lesion')
plt.axis('off')
plt.show()