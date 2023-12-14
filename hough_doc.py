import sys
import cv2 as cv
import numpy as np
from sklearn.cluster import DBSCAN

def detect_nodes(lines):
    intersections = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            rho1, theta1 = lines[i][0]
            rho2, theta2 = lines[j][0]
            A = np.array([
                [np.cos(theta1), np.sin(theta1)],
                [np.cos(theta2), np.sin(theta2)]
            ])
            determinant = np.linalg.det(A)
            if abs(determinant) > 1e-5:  # Checking if the determinant is not close to zero
                b = np.array([rho1, rho2])
                intersection = np.linalg.solve(A, b)
                if 0 <= intersection[0] < 1000 and 0 <= intersection[1] < 1000:
                    intersections.append([int(intersection[0]), int(intersection[1])])

    return np.array(intersections) if intersections else None

def find_significant_nodes(nodes):
    clustering = DBSCAN(eps=5, min_samples=3).fit(nodes)
    labels = clustering.labels_
    unique_labels = set(labels)
    
    significant_nodes = []
    for label in unique_labels:
        if label != -1:  # Excluding noise points
            cluster = nodes[labels == label]
            significant_node = np.mean(cluster, axis=0)
            significant_nodes.append(significant_node)
    
    return significant_nodes

def main(argv):
    default_file = "hand.png"
    filename = argv[0] if len(argv) > 0 else default_file

    # Loads an image
    src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_GRAYSCALE)

    # Check if image is loaded fine
    if src is None:
        print('Error opening image!')
        print('Usage: hough_lines.py [image_name -- default ' + default_file + '] \n')
        return -1

    # Edge detection
    dst = cv.Canny(src, 50, 200, None, 3)

    # Standard Hough Line Transform
    lines = cv.HoughLines(dst, 1, np.pi / 180, 200)

    # Draw the lines
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv.line(src, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)

    # Show only the lines for verification
    cv.imshow("Detected Lines", src)

    # Detect nodes
    if lines is not None:
        src_color = cv.cvtColor(src, cv.COLOR_GRAY2BGR)
        nodes = detect_nodes(lines)
        if nodes is not None:
            significant_nodes = find_significant_nodes(nodes)
            for point in significant_nodes:
                cv.circle(src_color, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)
                # Print coordinates of significant nodes
                print(f"Significant Node at: ({int(point[0])}, {int(point[1])})")

    # Show results
    cv.imshow("Detected Lines and Node Points", src_color)

    # Wait and Exit
    cv.waitKey()
    return 0

if __name__ == "__main__":
    main(sys.argv[1:])
