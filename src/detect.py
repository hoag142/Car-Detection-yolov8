#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Car Detection using YOLO

This script performs car detection on images, videos, or webcam feed using a trained YOLOv8 model.
It includes various image processing techniques to improve detection visualization.
"""

import os
import sys
import argparse
import time
import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Car Detection using YOLOv8')
    
    parser.add_argument('--source', type=str, default='0', 
                        help='Source for detection: file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--model', type=str, default='../models/best_car_detection.pt',
                        help='Path to the YOLOv8 model')
    parser.add_argument('--conf', type=float, default=0.25, 
                        help='Confidence threshold for detections')
    parser.add_argument('--iou', type=float, default=0.45, 
                        help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, 
                        help='Maximum detections per image')
    parser.add_argument('--view-img', action='store_true', 
                        help='Show detection results')
    parser.add_argument('--save-img', action='store_true', 
                        help='Save detection results')
    parser.add_argument('--save-dir', type=str, default='../outputs', 
                        help='Directory to save results')
    parser.add_argument('--classes', type=int, nargs='+', 
                        help='Filter by class: --classes 0, or --classes 0 1 2')
    parser.add_argument('--visualize', type=str, default='standard', 
                        choices=['standard', 'blur_bg', 'heatmap', 'edge'],
                        help='Visualization method')
    
    return parser.parse_args()


def apply_blur_background(img, boxes, blur_amount=21):
    """Apply blur to background, keeping detected objects sharp"""
    blurred_img = cv2.GaussianBlur(img, (blur_amount, blur_amount), 0)
    result_img = blurred_img.copy()
    
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        result_img[y1:y2, x1:x2] = img[y1:y2, x1:x2]
    
    return result_img


def apply_heatmap_overlay(img, boxes, confidences):
    """Create a heatmap overlay based on detection confidences"""
    heatmap = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    
    for box, conf in zip(boxes, confidences):
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(heatmap, (x1, y1), (x2, y2), float(conf), -1)
    
    # Normalize and apply colormap
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = np.uint8(heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Blend with original image
    return cv2.addWeighted(img, 0.7, heatmap, 0.3, 0)


def apply_edge_enhancement(img, boxes):
    """Enhance edges of detected objects"""
    result = img.copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Convert edges back to BGR
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # Create a mask for the detections
    mask = np.zeros_like(img)
    
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        # Expand the box slightly
        padding = 5
        x1_pad = max(0, x1 - padding)
        y1_pad = max(0, y1 - padding)
        x2_pad = min(img.shape[1], x2 + padding)
        y2_pad = min(img.shape[0], y2 + padding)
        
        # Add edges around the detection
        mask[y1_pad:y2_pad, x1_pad:x2_pad] = edges_bgr[y1_pad:y2_pad, x1_pad:x2_pad]
    
    # Overlay edges on original image
    result = cv2.addWeighted(result, 0.8, mask, 1, 0)
    
    return result


def process_frame(frame, model, args):
    """Process a single frame with the model and apply visualizations"""
    # Predict with YOLOv8
    results = model(frame, conf=args.conf, iou=args.iou, max_det=args.max_det, classes=args.classes)
    
    # Get detection results
    result = results[0]
    boxes = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    
    # Apply selected visualization method
    if args.visualize == 'standard':
        # Use the default YOLO visualization
        processed_frame = result.plot()
    elif args.visualize == 'blur_bg' and len(boxes) > 0:
        processed_frame = apply_blur_background(frame, boxes)
        # Draw boxes after applying blur
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            confidence = confidences[i]
            class_id = class_ids[i]
            
            # Get class name from model's names dictionary
            class_name = model.names[class_id]
            
            # Draw bounding box
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label
            label = f"{class_name} {confidence:.2f}"
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(processed_frame, (x1, y1-label_height-5), (x1+label_width, y1), (0, 255, 0), -1)
            cv2.putText(processed_frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    elif args.visualize == 'heatmap' and len(boxes) > 0:
        processed_frame = apply_heatmap_overlay(frame, boxes, confidences)
        # Draw boxes
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
    
    elif args.visualize == 'edge' and len(boxes) > 0:
        processed_frame = apply_edge_enhancement(frame, boxes)
        # Draw boxes
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            confidence = confidences[i]
            class_id = class_ids[i]
            
            # Get class name
            class_name = model.names[class_id]
            
            # Draw bounding box
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Add label
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(processed_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    else:
        # Fallback to standard visualization
        processed_frame = result.plot()
        
    return processed_frame, result


def process_image(image_path, model, args):
    """Process a single image file"""
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading image {image_path}")
        return
    
    # Process the image
    processed_img, result = process_frame(img, model, args)
    
    # Display if requested
    if args.view_img:
        cv2.imshow("Car Detection", processed_img)
        cv2.waitKey(0)
    
    # Save if requested
    if args.save_img:
        os.makedirs(args.save_dir, exist_ok=True)
        output_path = os.path.join(args.save_dir, Path(image_path).name)
        cv2.imwrite(output_path, processed_img)
        print(f"Result saved to {output_path}")


def process_video(video_path, model, args):
    """Process a video file or webcam feed"""
    # Open video source
    if video_path.isdigit():
        video_path = int(video_path)  # Use webcam
        print(f"Opening webcam {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video source {video_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Initialize video writer if saving
    out = None
    if args.save_img:
        os.makedirs(args.save_dir, exist_ok=True)
        if video_path == 0:  # webcam
            output_path = os.path.join(args.save_dir, f"webcam_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
        else:
            output_path = os.path.join(args.save_dir, f"{Path(video_path).stem}_detected.mp4")
        
        out = cv2.VideoWriter(output_path, 
                              cv2.VideoWriter_fourcc(*'mp4v'), 
                              fps, 
                              (width, height))
        print(f"Saving output to {output_path}")
    
    # Process video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process the frame
        processed_frame, result = process_frame(frame, model, args)
        
        # Display if requested
        if args.view_img:
            cv2.imshow("Car Detection", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
                break
        
        # Save if requested
        if args.save_img and out is not None:
            out.write(processed_frame)
    
    # Cleanup
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()


def main():
    """Main function"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Load the model
    try:
        model = YOLO(args.model)
        print(f"Model loaded: {args.model}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Process the source
    source = args.source
    
    if source.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
        # Process as image
        process_image(source, model, args)
    elif source.endswith(('.mp4', '.avi', '.mov', '.mkv')) or source.isdigit():
        # Process as video or webcam
        process_video(source, model, args)
    else:
        print(f"Unsupported source: {source}")


if __name__ == "__main__":
    main() 