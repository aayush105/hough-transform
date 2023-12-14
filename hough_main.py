import math
from sklearn.cluster import DBSCAN
import numpy as np
import cv2

def hough_transform(image, threshold, rho_resolution, theta_resolution):
    # Get image dimensions
    height, width = image.shape

    # Define the maximum possible distance in the image
    max_distance = int(math.sqrt(height ** 2 + width ** 2))

    # Define the accumulator array
    accumulator = np.zeros((2 * max_distance, theta_resolution), dtype=np.uint8)

    # Iterate over the image
    for y in range(height):
        for x in range(width):
            # Check if the pixel is an edge
            if image[y, x] != 0:
                # Iterate over all possible theta values
                for t_idx in range(theta_resolution):
                    theta = t_idx * (np.pi / theta_resolution)

                    # Calculate rho
                    rho = int(x * np.cos(theta) + y * np.sin(theta))

                    # Increment the accumulator cell
                    accumulator[rho + max_distance, t_idx] += 1

    # Find the lines based on the accumulator values
    lines = []
    for rho_idx in range(accumulator.shape[0]):
        for t_idx in range(accumulator.shape[1]):
            if accumulator[rho_idx, t_idx] > threshold:
                rho = rho_idx - max_distance
                theta = t_idx * (np.pi / theta_resolution)
                lines.append((rho, theta))

    return lines


def draw_lines(image, lines, width, height):
    for rho, theta in lines:
        # Convert polar coordinates to Cartesian coordinates
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho

        # Find two points on the line to draw a line segment
        x1 = int(x0 + width * (-b))
        y1 = int(y0 + height * (a))
        x2 = int(x0 - width * (-b))
        y2 = int(y0 - height * (a))

        # Draw the line on the image
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return image

def hough_transform_dbscan(image, threshold, rho_resolution, theta_resolution):
    lines = hough_transform(image, threshold, rho_resolution, theta_resolution)
    
    # DBSCAN clustering logic on the lines detected
    if len(lines) > 0:
        # Convert the lines to a numpy array
        lines_array = np.array(lines)
        
        # Perform DBSCAN clustering on the lines
        dbscan = DBSCAN(eps=2, min_samples=2)
        clusters = dbscan.fit_predict(lines)
        
        # Get the number of clusters found
        num_clusters = len(set(clusters))

        # Process each cluster
        best_lines = []
        for i in range(num_clusters):
            # Extract the lines in this cluster
            cluster = lines_array[clusters == i]
            
            if len(cluster) > 0:  # Check if the cluster is not empty
                # Find the line with the most votes in this cluster
                best_line = cluster[np.argmax(cluster[:, 0]), :]
                best_lines.append(best_line)
        
        if len(best_lines) > 0:  # Check if any valid lines were found
            return best_lines
        else:
            return None  # Return None if no valid lines were found after clustering
            
    else:
        return None  # Return None if no lines were detected by the initial Hough Transform


def draw_lines_dbscan(image, lines):
    if lines is not None and len(lines) > 0:
        for rho, theta in lines:
            # Convert polar coordinates to Cartesian coordinates
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho

            # Find two points on the line to draw a line segment
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            # Draw the line on the image
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return image

def main():
    # Load the image
    image = cv2.imread('Project/ckt4.jpg')

    # Convert the image to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    image_edges = cv2.Canny(image_gray, 100, 200)

    # Get image dimensions
    height, width = image_gray.shape

    # Perform Hough Transform
    threshold = 90  # Minimum number of votes to consider a line
    rho_resolution = 1  # Resolution of the rho value
    theta_resolution = 180 # Resolution of the theta value
    lines = hough_transform(image_edges, threshold, rho_resolution, theta_resolution)

    # Draw the lines on the image
    image_lines = draw_lines(image.copy(), lines, width, height)

    # Perform Hough Transform using DBSCAN
    lines_dbscan = hough_transform_dbscan(image_edges, threshold, rho_resolution, theta_resolution)

    # Draw the lines detected by DBSCAN Hough Transform
    image_lines_dbscan = draw_lines_dbscan(image.copy(), lines_dbscan)

    # Find intersection points among the lines detected by DBSCAN
    intersection_points = []
    if lines_dbscan:  # Check if lines_dbscan is not empty or None
        for i in range(len(lines_dbscan)):
            for j in range(i + 1, len(lines_dbscan)):
                rho1, theta1 = lines_dbscan[i]
                rho2, theta2 = lines_dbscan[j]

                # Check if lines are not parallel
                if abs(theta1 - theta2) > 0.01:  # Adjust the threshold as needed
                    # Calculate the intersection point between lines
                    A = np.array([
                        [np.cos(theta1), np.sin(theta1)],
                        [np.cos(theta2), np.sin(theta2)]
                    ])
                    b = np.array([rho1, rho2])
                    intersection = np.linalg.solve(A, b)
                    intersection_points.append(intersection)
                else:
                    # Lines are close to parallel, handle this case as needed
                    pass  # For example, skip calculating intersection for parallel lines
    else:
        print("No lines detected by DBSCAN")

    # Find and print the coordinates of intersection points
    intersection_coordinates = []
    for point in intersection_points:
        x, y = point
        intersection_coordinates.append((x, y))
        print(f"Intersection Point Coordinates: ({x}, {y})")

    # Draw the lines detected by DBSCAN Hough Transform
    image_lines_dbscan_with_intersections = draw_lines_dbscan(image.copy(), lines_dbscan)

    # Draw intersection points on the image
    for point in intersection_points:
        x, y = point
        cv2.circle(image_lines_dbscan_with_intersections, (int(x), int(y)), 5, (255, 0, 0), -1)

    # Display the images with lines and intersection points
    cv2.imshow('Hough Transform (DBSCAN) with Intersections', image_lines_dbscan_with_intersections)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()



    


    