import cv2
import sys
import numpy as np
import math
from sklearn.cluster import DBSCAN

import numpy as np

def find_intersection_points(lines):
    intersection_points = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            line1 = lines[i]
            line2 = lines[j]
            denominator = (
                (line2[1][1] - line2[0][1]) * (line1[1][0] - line1[0][0])
                - (line2[1][0] - line2[0][0]) * (line1[1][1] - line1[0][1])
            )
            if denominator != 0:  # Checking if lines are not almost parallel
                intersection = [
                    (
                        (line2[1][0] - line2[0][0])
                        * (line1[1][0] * line1[0][1] - line1[0][0] * line1[1][1])
                        - (line1[1][0] - line1[0][0])
                        * (line2[1][0] * line2[0][1] - line2[0][0] * line2[1][1])
                    )
                    / denominator,
                    (
                        (line1[0][1] - line1[1][1])
                        * (line2[1][0] * line2[0][1] - line2[0][0] * line2[1][1])
                        - (line2[0][1] - line2[1][1])
                        * (line1[1][0] * line1[0][1] - line1[0][0] * line1[1][1])
                    )
                    / denominator,
                ]
                intersection = np.array(intersection)
                if np.all(0 <= intersection) and np.all(intersection <= 1000):
                    intersection_points.append(intersection.astype(int))

    return np.array(intersection_points) if intersection_points else None


def find_significant_nodes(intersection_points):
    clustering = DBSCAN(eps=8, min_samples=4).fit(intersection_points)
    labels = clustering.labels_
    unique_labels = set(labels)
    
    significant_nodes = []
    for label in unique_labels:
        if label != -1:  # Excluding noise points
            cluster = intersection_points[labels == label]
            significant_node = np.mean(cluster, axis=0)
            significant_nodes.append(significant_node)
    
    return significant_nodes

# Canny edge detection function
def canny_edge_detection(image, threshold1, threshold2):
    # Grayscale conversion
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gaussian Blur
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Canny edge detection
    edges = cv2.Canny(blurred, threshold1, threshold2)
    return edges

# Probabilistic Hough Transform function
def probabilistic_hough_transform(image_edges, threshold, min_line_length, max_line_gap):
    lines = []  # Store detected lines [(x1, y1, x2, y2), ...]

    height, width = image_edges.shape[:2]

    # Voting space parameters
    rho_max = int(math.hypot(height, width))
    theta_max = 180
    accumulator = np.zeros((2 * rho_max, theta_max), dtype=np.uint8)

    edge_points = np.argwhere(image_edges > 0)  # Get edge points

    # Perform voting for lines
    for y, x in edge_points:
        for theta in range(theta_max):
            rho = int(x * math.cos(math.radians(theta)) + y * math.sin(math.radians(theta)))
            accumulator[rho, theta] += 1  # Accumulate votes

    # Threshold accumulator to find lines
    for rho in range(accumulator.shape[0]):
        for theta in range(accumulator.shape[1]):
            if accumulator[rho, theta] > threshold:
                a = math.cos(math.radians(theta))
                b = math.sin(math.radians(theta))
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                lines.append(((x1, y1), (x2, y2)))  # Store line endpoints

    return lines

def main(argv):
    # Load the input image
    image = cv2.imread('Project/hand.png')

    edges = canny_edge_detection(image, 50, 150)

    # Perform Probabilistic Hough Transform
    threshold = 100  # Minimum number of votes to consider a line
    min_line_length = 50  # Minimum length of a line
    max_line_gap = 10  # Maximum allowed gap between line segments to treat them as a single line

    lines = probabilistic_hough_transform(edges, threshold, min_line_length, max_line_gap)

    # Find intersection points from detected lines
    intersection_points = find_intersection_points(lines)

    # Find significant nodes using DBSCAN
    significant_nodes = find_significant_nodes(intersection_points)

    
    display_image = image.copy()

    # Draw detected lines
    for line in lines:
        cv2.line(display_image, line[0], line[1], (0, 255, 0), 2)  

    # Draw intersection points
    if intersection_points is not None:
        for point in intersection_points:
            cv2.circle(display_image, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)  

    # Draw significant nodes (with smaller radius)
    if significant_nodes:
        for node in significant_nodes:
            cv2.circle(display_image, (int(node[0]), int(node[1])), 3, (255, 0, 0), -1)  
            print(f"Significant Node at: ({int(node[0])}, {int(node[1])})")

    # Display the combined image
    cv2.imshow('Detected nodes and lines', display_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(sys.argv[1:])

