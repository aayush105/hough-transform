import sys
import cv2 as cv
import numpy as np
from sklearn.cluster import DBSCAN

def detect_nodes(lines):
    intersections = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            x1, y1, x2, y2 = lines[i][0]
            x3, y3, x4, y4 = lines[j][0]
            denominator = ((y4 - y3) * (x2 - x1)) - ((x4 - x3) * (y2 - y1))
            if denominator != 0:  # Checking if lines are not almost parallel
                intersection = [((x4 - x3) * (x2 * y1 - x1 * y2) - (x2 - x1) * (x4 * y3 - x3 * y4)) / denominator,
                                ((y1 - y2) * (x4 * y3 - x3 * y4) - (y3 - y4) * (x2 * y1 - x1 * y2)) / denominator]
                if 0 <= intersection[0] < 1000 and 0 <= intersection[1] < 1000:
                    intersections.append([int(intersection[0]), int(intersection[1])])

    return np.array(intersections) if intersections else None

def find_significant_nodes(nodes):
    clustering = DBSCAN(eps=10, min_samples=2).fit(nodes)
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
    default_file = "Project/ckt5.jpeg"
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

    # Probabilistic Hough Line Transform
    lines = cv.HoughLinesP(dst, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)

    # Draw the lines
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv.line(src, (x1, y1), (x2, y2), (0, 0, 255), 3, cv.LINE_AA)

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