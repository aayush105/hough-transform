import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from scipy import ndimage


def canny_edge_detection(image, low_threshold, high_threshold):
    blurred_image = ndimage.gaussian_filter(image, sigma=1.4)

    gradient_x = ndimage.sobel(blurred_image, axis=1)
    gradient_y = ndimage.sobel(blurred_image, axis=0)

    gradient_magnitude = np.hypot(gradient_x, gradient_y)
    gradient_direction = np.arctan2(gradient_y, gradient_x)

    gradient_direction = np.rad2deg(gradient_direction) % 180

    edges = np.zeros_like(image)
    edges[(gradient_magnitude >= low_threshold) & (gradient_magnitude <= high_threshold)] = 255

    return edges


def hough_transform(image, theta_res=1, rho_res=1):
    height, width = image.shape
    diag_len = int(np.ceil(np.sqrt(height**2 + width**2)))  # Maximum possible rho value
    thetas = np.deg2rad(np.arange(-90, 90, theta_res))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2 * rho_res)

    cos_thetas = np.cos(thetas)
    sin_thetas = np.sin(thetas)
    num_thetas = len(thetas)

    accumulator = np.zeros((2 * diag_len * rho_res, num_thetas), dtype=np.uint64)

    y_indices, x_indices = np.nonzero(image)
    for i in range(len(x_indices)):
        x = x_indices[i]
        y = y_indices[i]
        for theta_idx in range(num_thetas):
            rho = int(round(x * cos_thetas[theta_idx] + y * sin_thetas[theta_idx]) + diag_len * rho_res)
            accumulator[rho, theta_idx] += 1

    return accumulator, thetas, rhos


def find_lines(accumulator, thetas, rhos, threshold):
    lines = []
    peaks = np.where(accumulator >= threshold)
    for rho, theta_idx in zip(peaks[0], peaks[1]):
        rho_val = rhos[rho]
        theta_val = thetas[theta_idx]
        lines.append((rho_val, theta_val))
    return lines


def draw_lines(image, lines):
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    width, height = img.size

    # Set background color to white
    draw.rectangle([(0, 0), (width, height)], fill=255)

    # Set line color to black
    line_color = 0

    # Adjust line thickness based on image scale
    line_thickness = int(np.sqrt(width**2 + height**2) * 0.005)

    for rho, theta in lines:
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        x0 = rho * cos_theta
        y0 = rho * sin_theta
        x1 = int(x0 + 1000 * (-sin_theta))
        y1 = int(y0 + 1000 * (cos_theta))
        x2 = int(x0 - 1000 * (-sin_theta))
        y2 = int(y0 - 1000 * (cos_theta))

        # Draw a single line segment
        draw.line([(x1, y1), (x2, y2)], fill=line_color, width=line_thickness)
    return img





# Load the image
image = np.array(Image.open('ckt.jpg').convert('L'))

# Apply Canny edge detection
low_threshold = 50
high_threshold = 150
edge_image = canny_edge_detection(image, low_threshold, high_threshold)

# Run Hough Transform
accumulator, thetas, rhos = hough_transform(edge_image)

# Find lines
threshold = 50
lines = find_lines(accumulator, thetas, rhos, threshold)

# Check if any lines were detected
if len(lines) > 0:
    # Draw lines on the original image
    output_image = draw_lines(image, lines)

    # Display the output
    plt.imshow(output_image, cmap='gray')
    plt.axis('off')
    plt.show()
else:
    print("No lines detected.")
