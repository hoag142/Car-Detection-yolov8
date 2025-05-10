#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Image Processing Utilities for Car Detection

This module provides various image processing techniques that can be used 
for pre-processing, post-processing, and visualization of car detection results.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def apply_histogram_equalization(image):
    """
    Apply histogram equalization to enhance image contrast.
    
    Args:
        image (numpy.ndarray): Input image in BGR format
        
    Returns:
        numpy.ndarray: Image with enhanced contrast
    """
    # Convert to YUV color space
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    
    # Apply histogram equalization to Y channel
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    
    # Convert back to BGR
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)


def apply_clahe(image):
    """
    Apply Contrast Limited Adaptive Histogram Equalization for better contrast.
    
    Args:
        image (numpy.ndarray): Input image in BGR format
        
    Returns:
        numpy.ndarray: Image with enhanced local contrast
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Split the channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Merge channels and convert back to BGR
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def denoise_image(image, method='gaussian', strength=5):
    """
    Apply noise reduction to the image.
    
    Args:
        image (numpy.ndarray): Input image in BGR format
        method (str): Denoising method ('gaussian', 'median', 'bilateral', 'nlm')
        strength (int): Strength of denoising effect
    
    Returns:
        numpy.ndarray: Denoised image
    """
    if method == 'gaussian':
        return cv2.GaussianBlur(image, (strength, strength), 0)
    
    elif method == 'median':
        return cv2.medianBlur(image, strength)
    
    elif method == 'bilateral':
        return cv2.bilateralFilter(image, strength, 75, 75)
    
    elif method == 'nlm':  # Non-local means
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    
    return image


def sharpen_image(image, kernel_size=3, strength=1.0):
    """
    Sharpen an image using unsharp masking.
    
    Args:
        image (numpy.ndarray): Input image in BGR format
        kernel_size (int): Size of the Gaussian kernel
        strength (float): Strength of sharpening effect
    
    Returns:
        numpy.ndarray: Sharpened image
    """
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    # Calculate unsharp mask
    unsharp_mask = cv2.addWeighted(image, 1.0 + strength, blur, -strength, 0)
    
    # Ensure pixel values are within range
    return np.clip(unsharp_mask, 0, 255).astype(np.uint8)


def detect_edges(image, low_threshold=50, high_threshold=150):
    """
    Detect edges in an image using Canny edge detector.
    
    Args:
        image (numpy.ndarray): Input image in BGR format
        low_threshold (int): Lower threshold for the hysteresis procedure
        high_threshold (int): Upper threshold for the hysteresis procedure
    
    Returns:
        numpy.ndarray: Binary edge map
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detect edges
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    
    return edges


def apply_morphological_operations(image, operation='dilate', kernel_size=5, iterations=1):
    """
    Apply morphological operations to an image.
    
    Args:
        image (numpy.ndarray): Input binary image
        operation (str): Type of operation ('dilate', 'erode', 'open', 'close')
        kernel_size (int): Size of the structuring element
        iterations (int): Number of times to apply the operation
    
    Returns:
        numpy.ndarray: Processed image
    """
    # Create kernel
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    if operation == 'dilate':
        return cv2.dilate(image, kernel, iterations=iterations)
    
    elif operation == 'erode':
        return cv2.erode(image, kernel, iterations=iterations)
    
    elif operation == 'open':
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    
    elif operation == 'close':
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    
    return image


def highlight_detection_area(image, boxes, color=(0, 255, 0), alpha=0.3):
    """
    Highlight the detection areas with semi-transparent overlay.
    
    Args:
        image (numpy.ndarray): Input image in BGR format
        boxes (list): List of bounding boxes in format [x1, y1, x2, y2]
        color (tuple): RGB color for the highlight
        alpha (float): Transparency factor (0 to 1)
    
    Returns:
        numpy.ndarray: Image with highlighted detection areas
    """
    # Create an overlay
    overlay = image.copy()
    
    # Draw filled rectangles on the overlay
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    
    # Blend with original image
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


def apply_color_filter(image, color_lower, color_upper):
    """
    Apply color filtering to extract specific color ranges.
    
    Args:
        image (numpy.ndarray): Input image in BGR format
        color_lower (tuple): Lower bounds for color in HSV
        color_upper (tuple): Upper bounds for color in HSV
    
    Returns:
        numpy.ndarray: Binary mask of the filtered color regions
    """
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Create a mask for the specified color range
    mask = cv2.inRange(hsv, np.array(color_lower), np.array(color_upper))
    
    return mask


def enhance_car_detection(image, boxes):
    """
    Enhance car detection visualization with multiple techniques.
    
    Args:
        image (numpy.ndarray): Input image in BGR format
        boxes (list): List of bounding boxes in format [x1, y1, x2, y2]
    
    Returns:
        numpy.ndarray: Enhanced visualization
    """
    # Create a copy for processing
    result = image.copy()
    
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # Create a mask for detected regions with some padding
    mask = np.zeros_like(image)
    
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        # Add some padding
        pad = 20
        x1_pad = max(0, x1 - pad)
        y1_pad = max(0, y1 - pad)
        x2_pad = min(image.shape[1], x2 + pad)
        y2_pad = min(image.shape[0], y2 + pad)
        
        # Extract car region with padding
        car_region = image[y1_pad:y2_pad, x1_pad:x2_pad]
        
        if car_region.size > 0:
            # Enhance contrast for car region
            enhanced = apply_clahe(car_region)
            
            # Sharpen the car region
            sharpened = sharpen_image(enhanced, kernel_size=3, strength=1.5)
            
            # Replace in the result image
            result[y1_pad:y2_pad, x1_pad:x2_pad] = sharpened
            
            # Add edge highlight
            mask[y1_pad:y2_pad, x1_pad:x2_pad] = edges_color[y1_pad:y2_pad, x1_pad:x2_pad]
    
    # Overlay edges on the result
    result = cv2.addWeighted(result, 0.8, mask, 0.7, 0)
    
    # Highlight detection areas
    result = highlight_detection_area(result, boxes, color=(0, 255, 0), alpha=0.15)
    
    return result


def visualize_multiple_processing(image):
    """
    Visualize multiple image processing techniques on a single image.
    
    Args:
        image (numpy.ndarray): Input image in BGR format
        
    Returns:
        numpy.ndarray: Visualization grid with multiple processed versions
    """
    # Convert BGR to RGB for matplotlib
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply various processing techniques
    hist_eq = apply_histogram_equalization(image)
    hist_eq_rgb = cv2.cvtColor(hist_eq, cv2.COLOR_BGR2RGB)
    
    clahe_img = apply_clahe(image)
    clahe_rgb = cv2.cvtColor(clahe_img, cv2.COLOR_BGR2RGB)
    
    denoised = denoise_image(image, method='bilateral')
    denoised_rgb = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)
    
    sharpened = sharpen_image(image)
    sharpened_rgb = cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)
    
    edges = detect_edges(image)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    
    # Convert to grayscale and apply threshold
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
    
    # Create a 3x3 grid for visualization
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    # Display images
    axes[0, 0].imshow(rgb_img)
    axes[0, 0].set_title('Original')
    
    axes[0, 1].imshow(hist_eq_rgb)
    axes[0, 1].set_title('Histogram Equalization')
    
    axes[0, 2].imshow(clahe_rgb)
    axes[0, 2].set_title('CLAHE')
    
    axes[1, 0].imshow(denoised_rgb)
    axes[1, 0].set_title('Denoised (Bilateral)')
    
    axes[1, 1].imshow(sharpened_rgb)
    axes[1, 1].set_title('Sharpened')
    
    axes[1, 2].imshow(edges_rgb)
    axes[1, 2].set_title('Edge Detection')
    
    axes[2, 0].imshow(thresh_rgb)
    axes[2, 0].set_title('Threshold (Otsu)')
    
    # HSV color space visualization
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    hsv_display = np.concatenate((h, s, v), axis=1)
    axes[2, 1].imshow(hsv_display, cmap='gray')
    axes[2, 1].set_title('HSV Channels (H, S, V)')
    
    # Turn off empty subplot
    axes[2, 2].axis('off')
    
    # Turn off axis ticks
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    
    # Convert the plot to an image
    fig.canvas.draw()
    plot_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    
    return plot_img 