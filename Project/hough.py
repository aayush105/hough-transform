import cv2
import numpy as np
from sklearn.cluster import KMeans
import math
import sys 

def find_intersection(line1, line2):
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    
    # Check if the lines are parallel
    if np.linalg.cond(A) < 1/sys.float_info.epsilon:
        x0, y0 = np.linalg.solve(A, b)
        x0, y0 = int(np.round(x0)), int(np.round(y0))
        return (x0, y0)
    else:
        return None  # return None if the lines are parallel


# Load the input image
image = cv2.imread('ckt1.jpg', 0)

# Apply Gaussian blur
image = cv2.GaussianBlur(image, (5, 5), 0)

# Apply Canny edge detection
edges = cv2.Canny(image, 50, 150, apertureSize=3)

# Perform Hough Transform
lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

# Create a copy of the original image to draw lines and points
image_copy = image.copy()

# Draw the lines
for rho, theta in lines[:, 0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv2.line(image_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)

intersection_points = []
for i, line1 in enumerate(lines):
    for line2 in lines[i+1:]:
        intersection = find_intersection(line1, line2)
        if intersection is not None:  # check if the lines intersect
            intersection_points.append(intersection)

# Draw the intersection points
for point in intersection_points:
    cv2.circle(image_copy, point, 5, (255, 0, 0), -1)

if intersection_points:  # check if intersection_points is not empty
    kmeans = KMeans(n_clusters=min(8, len(intersection_points)), random_state=0).fit(intersection_points)
    cluster_centers = kmeans.cluster_centers_

    # Draw the node points
    for point in kmeans.cluster_centers_:
        cv2.circle(image_copy, (int(point[0]), int(point[1])), 10, (0, 255, 0), -1)
else:
    print("No intersection points found.")

# Perform KMeans clustering
# kmeans = KMeans(n_clusters=min(8, len(intersection_points)), random_state=0).fit(intersection_points)


# Display the image with lines and points
cv2.imshow('Image with Lines and Points', image_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()