import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

def rgb2gray(rgb):
    # Convert an RGB image to grayscale
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

def show_hough_lines_cv(lines, save_path=None):
    """
    Create a scatter plot of detected Hough lines.
    lines - array of shape (N, 2) with each line as [rho, theta] (theta in radians)
    """
    # Separate rho and theta values
    rhos = lines[:, 0]
    thetas = lines[:, 1]
    
    # Plot theta (converted to degrees) vs rho
    plt.figure(figsize=(8, 6))
    plt.scatter(np.rad2deg(thetas), rhos, c='red', marker='o')
    plt.xlabel("Theta (degrees)")
    plt.ylabel("Rho (pixels)")
    plt.title("Detected Lines in Hough Space")
    # Invert y-axis so larger rho values appear at the bottom
    plt.gca().invert_yaxis()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    imgpath = "writing/work/imgs/cv/methodik/edges_gray.jpg"
    img = cv2.imread(imgpath)
    if img is None:
        raise IOError("Image not found at path: " + imgpath)
    if img.ndim == 3:
        img = rgb2gray(img)
    img = cv2.resize(img, (400, 400))
    
    # Optionally, run edge detection if the image is not already binary.
    # Here, we use Canny to generate a binary edge image.
    edges = cv2.Canny(img, 50, 150)
    
    # Use OpenCV's HoughLines:
    # Parameters: image, rho resolution, theta resolution, threshold
    # Adjust the threshold (e.g., 100) depending on the edge image quality.
    lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
    
    if lines is not None:
        # Reshape the result to a simple (N,2) array.
        lines = lines[:, 0, :]  # each line is [rho, theta]
        # (Optional) Sort lines by rho or by the vote value if available.
        show_hough_lines_cv(lines, save_path="output.png")
    else:
        print("No lines detected.")
