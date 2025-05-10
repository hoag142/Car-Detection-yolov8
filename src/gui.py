#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Car Detection GUI Application

This script provides a graphical user interface for the car detection system,
allowing users to detect cars in images, videos, or webcam feeds.
"""

import os
import sys
import time
import cv2
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
import threading
from pathlib import Path

# Add parent directory to path to import from project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from ultralytics import YOLO
from utils.image_processing import (
    apply_clahe, sharpen_image, enhance_car_detection,
    visualize_multiple_processing
)


class CarDetectionApp:
    """Car Detection GUI Application"""
    
    def __init__(self, root):
        """Initialize the application"""
        self.root = root
        self.root.title("Car Detection using YOLO")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)
        
        # Set app icon if available
        try:
            self.root.iconbitmap("../assets/car_icon.ico")
        except:
            pass
        
        # Variables
        self.model = None
        self.model_path = tk.StringVar(value="../models/best_car_detection.pt")
        self.confidence = tk.DoubleVar(value=0.25)
        self.source_path = tk.StringVar()
        self.visualization_mode = tk.StringVar(value="standard")
        
        self.cap = None  # Video capture
        self.is_webcam = False
        self.is_video = False
        self.is_processing = False
        self.stop_processing = False
        
        # Create UI
        self.create_ui()
        
        # Load model
        self.load_model()
    
    def create_ui(self):
        """Create the user interface"""
        # Main layout
        self.main_frame = ttk.Frame(self.root, padding=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel (controls)
        self.left_panel = ttk.Frame(self.main_frame, padding=10, width=300)
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y, expand=False)
        
        # Model section
        model_frame = ttk.LabelFrame(self.left_panel, text="Model", padding=10)
        model_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(model_frame, text="Model Path:").pack(anchor=tk.W)
        model_path_frame = ttk.Frame(model_frame)
        model_path_frame.pack(fill=tk.X, pady=5)
        ttk.Entry(model_path_frame, textvariable=self.model_path).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(model_path_frame, text="Browse", command=self.browse_model).pack(side=tk.RIGHT)
        ttk.Button(model_frame, text="Load Model", command=self.load_model).pack(fill=tk.X)
        
        # Detection settings
        settings_frame = ttk.LabelFrame(self.left_panel, text="Detection Settings", padding=10)
        settings_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(settings_frame, text="Confidence Threshold:").pack(anchor=tk.W)
        conf_scale = ttk.Scale(settings_frame, from_=0.01, to=0.99, variable=self.confidence, length=200)
        conf_scale.pack(fill=tk.X)
        ttk.Label(settings_frame, textvariable=tk.StringVar(value=lambda: f"{self.confidence.get():.2f}")).pack()
        
        ttk.Label(settings_frame, text="Visualization Mode:").pack(anchor=tk.W, pady=(10, 0))
        modes = ["standard", "blur_bg", "heatmap", "edge"]
        vis_combo = ttk.Combobox(settings_frame, textvariable=self.visualization_mode, values=modes)
        vis_combo.pack(fill=tk.X, pady=5)
        
        # Source section
        source_frame = ttk.LabelFrame(self.left_panel, text="Source", padding=10)
        source_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(source_frame, text="Open Image", command=self.open_image).pack(fill=tk.X, pady=2)
        ttk.Button(source_frame, text="Open Video", command=self.open_video).pack(fill=tk.X, pady=2)
        ttk.Button(source_frame, text="Open Webcam", command=self.open_webcam).pack(fill=tk.X, pady=2)
        ttk.Button(source_frame, text="Stop Processing", command=self.stop).pack(fill=tk.X, pady=2)
        ttk.Button(source_frame, text="Save Current Frame", command=self.save_frame).pack(fill=tk.X, pady=2)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.left_panel, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM, pady=5)
        
        # Right panel (display area)
        self.right_panel = ttk.Frame(self.main_frame)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Canvas for displaying results
        self.canvas_frame = ttk.Frame(self.right_panel)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.right_panel, orient=tk.HORIZONTAL, mode='indeterminate')
        self.progress.pack(fill=tk.X, side=tk.BOTTOM)
        
        # Bind window resize event
        self.root.bind("<Configure>", self.on_resize)
    
    def browse_model(self):
        """Browse for a model file"""
        model_path = filedialog.askopenfilename(
            title="Select YOLO Model",
            filetypes=[("PyTorch Model", "*.pt"), ("ONNX Model", "*.onnx"), ("All Files", "*.*")]
        )
        
        if model_path:
            self.model_path.set(model_path)
    
    def load_model(self):
        """Load the YOLO model"""
        model_path = self.model_path.get()
        
        if not os.path.exists(model_path):
            messagebox.showerror("Error", f"Model not found at: {model_path}")
            return
        
        self.status_var.set("Loading model...")
        self.progress.start()
        
        def load_model_thread():
            try:
                self.model = YOLO(model_path)
                self.root.after(0, lambda: self.status_var.set(f"Model loaded: {os.path.basename(model_path)}"))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to load model: {e}"))
                self.root.after(0, lambda: self.status_var.set("Model loading failed"))
            finally:
                self.root.after(0, self.progress.stop)
        
        threading.Thread(target=load_model_thread).start()
    
    def open_image(self):
        """Open an image file for detection"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp"), ("All Files", "*.*")]
        )
        
        if not file_path:
            return
        
        self.source_path.set(file_path)
        self.stop_processing = True
        self.is_video = False
        self.is_webcam = False
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        self.status_var.set(f"Processing image: {os.path.basename(file_path)}")
        self.progress.start()
        
        def process_image_thread():
            try:
                # Read image
                img = cv2.imread(file_path)
                if img is None:
                    self.root.after(0, lambda: messagebox.showerror("Error", "Failed to load image"))
                    return
                
                # Process image
                if self.model is None:
                    self.root.after(0, lambda: messagebox.showinfo("Info", "No model loaded. Loading default..."))
                    self.load_model()
                
                results = self.model(img, conf=self.confidence.get())
                result = results[0]
                
                # Apply visualization based on mode
                vis_mode = self.visualization_mode.get()
                if vis_mode == "standard":
                    processed_img = result.plot()
                elif vis_mode == "blur_bg":
                    boxes = result.boxes.xyxy.cpu().numpy()
                    if len(boxes) > 0:
                        # Apply blur background effect
                        blurred = cv2.GaussianBlur(img, (21, 21), 0)
                        processed_img = blurred.copy()
                        
                        for box in boxes:
                            x1, y1, x2, y2 = map(int, box[:4])
                            processed_img[y1:y2, x1:x2] = img[y1:y2, x1:x2]
                        
                        # Draw boxes
                        for i, box in enumerate(boxes):
                            x1, y1, x2, y2 = map(int, box[:4])
                            conf = result.boxes.conf[i].item()
                            cls = int(result.boxes.cls[i].item())
                            label = f"{result.names[cls]} {conf:.2f}"
                            
                            cv2.rectangle(processed_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(processed_img, label, (x1, y1-10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    else:
                        processed_img = img.copy()
                elif vis_mode == "heatmap":
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    
                    # Create heatmap
                    heatmap = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
                    for box, conf in zip(boxes, confidences):
                        x1, y1, x2, y2 = map(int, box[:4])
                        cv2.rectangle(heatmap, (x1, y1), (x2, y2), float(conf), -1)
                    
                    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
                    heatmap = np.uint8(heatmap)
                    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                    
                    # Blend with original
                    processed_img = cv2.addWeighted(img, 0.7, heatmap, 0.3, 0)
                    
                    # Draw boxes
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = map(int, box[:4])
                        cv2.rectangle(processed_img, (x1, y1), (x2, y2), (255, 255, 255), 2)
                elif vis_mode == "edge":
                    boxes = result.boxes.xyxy.cpu().numpy()
                    processed_img = enhance_car_detection(img, boxes)
                else:
                    processed_img = result.plot()
                
                # Display
                self.display_image(processed_img)
                
                # Update status
                n_objects = len(result.boxes)
                self.root.after(0, lambda: self.status_var.set(f"Found {n_objects} objects | {os.path.basename(file_path)}"))
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Processing failed: {e}"))
                self.root.after(0, lambda: self.status_var.set("Processing failed"))
            finally:
                self.root.after(0, self.progress.stop)
        
        threading.Thread(target=process_image_thread).start()
    
    def open_video(self):
        """Open a video file for detection"""
        file_path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")]
        )
        
        if not file_path:
            return
        
        self.source_path.set(file_path)
        self.stop_processing = True
        time.sleep(0.5)  # Wait for any ongoing processing to stop
        
        self.is_video = True
        self.is_webcam = False
        
        if self.cap is not None:
            self.cap.release()
        
        # Open video
        self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Failed to open video file")
            return
        
        self.status_var.set(f"Processing video: {os.path.basename(file_path)}")
        self.process_video()
    
    def open_webcam(self):
        """Open webcam for detection"""
        self.stop_processing = True
        time.sleep(0.5)  # Wait for any ongoing processing to stop
        
        self.is_webcam = True
        self.is_video = True
        
        if self.cap is not None:
            self.cap.release()
        
        # Open webcam (device 0)
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Failed to open webcam")
            return
        
        self.status_var.set("Processing webcam feed")
        self.process_video()
    
    def process_video(self):
        """Process video or webcam feed"""
        if self.model is None:
            messagebox.showinfo("Info", "No model loaded. Loading default...")
            self.load_model()
        
        self.stop_processing = False
        self.is_processing = True
        
        def process_video_thread():
            try:
                while self.cap.isOpened() and not self.stop_processing:
                    ret, frame = self.cap.read()
                    if not ret:
                        if self.is_webcam:  # Try to reconnect to webcam
                            self.cap.release()
                            self.cap = cv2.VideoCapture(0)
                            continue
                        else:
                            break
                    
                    # Process frame
                    results = self.model(frame, conf=self.confidence.get())
                    result = results[0]
                    
                    # Apply visualization based on mode
                    vis_mode = self.visualization_mode.get()
                    if vis_mode == "standard":
                        processed_frame = result.plot()
                    elif vis_mode == "blur_bg":
                        boxes = result.boxes.xyxy.cpu().numpy()
                        if len(boxes) > 0:
                            blurred = cv2.GaussianBlur(frame, (21, 21), 0)
                            processed_frame = blurred.copy()
                            
                            for box in boxes:
                                x1, y1, x2, y2 = map(int, box[:4])
                                processed_frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
                            
                            # Draw boxes
                            for i, box in enumerate(boxes):
                                x1, y1, x2, y2 = map(int, box[:4])
                                conf = result.boxes.conf[i].item()
                                cls = int(result.boxes.cls[i].item())
                                label = f"{result.names[cls]} {conf:.2f}"
                                
                                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(processed_frame, label, (x1, y1-10), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        else:
                            processed_frame = frame.copy()
                    elif vis_mode == "heatmap":
                        boxes = result.boxes.xyxy.cpu().numpy()
                        confidences = result.boxes.conf.cpu().numpy()
                        
                        # Create heatmap
                        heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
                        for box, conf in zip(boxes, confidences):
                            x1, y1, x2, y2 = map(int, box[:4])
                            cv2.rectangle(heatmap, (x1, y1), (x2, y2), float(conf), -1)
                        
                        heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
                        heatmap = np.uint8(heatmap)
                        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                        
                        # Blend with original
                        processed_frame = cv2.addWeighted(frame, 0.7, heatmap, 0.3, 0)
                        
                        # Draw boxes
                        for i, box in enumerate(boxes):
                            x1, y1, x2, y2 = map(int, box[:4])
                            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                    elif vis_mode == "edge":
                        boxes = result.boxes.xyxy.cpu().numpy()
                        processed_frame = enhance_car_detection(frame, boxes)
                    else:
                        processed_frame = result.plot()
                    
                    # Display
                    self.display_image(processed_frame)
                    
                    # Update status
                    n_objects = len(result.boxes)
                    source_name = "Webcam" if self.is_webcam else os.path.basename(self.source_path.get())
                    self.root.after(0, lambda: self.status_var.set(f"Found {n_objects} objects | {source_name}"))
                    
                    # Control frame rate
                    if self.is_webcam:
                        time.sleep(0.01)  # Prevent high CPU usage
                    else:
                        time.sleep(1/30)  # Cap at ~30 fps
            
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Processing failed: {e}"))
                
            finally:
                self.is_processing = False
                self.root.after(0, lambda: self.status_var.set("Ready"))
        
        threading.Thread(target=process_video_thread).start()
    
    def stop(self):
        """Stop current processing"""
        self.stop_processing = True
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        self.status_var.set("Processing stopped")
    
    def save_frame(self):
        """Save the current frame to a file"""
        if not hasattr(self, 'current_frame') or self.current_frame is None:
            messagebox.showinfo("Info", "No image to save")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Image",
            defaultextension=".jpg",
            filetypes=[("JPEG Image", "*.jpg"), ("PNG Image", "*.png"), ("All Files", "*.*")]
        )
        
        if file_path:
            try:
                cv2.imwrite(file_path, self.current_frame)
                self.status_var.set(f"Image saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image: {e}")
    
    def display_image(self, img):
        """Display an image on the canvas"""
        self.current_frame = img.copy()  # Store the current frame
        
        # Convert to RGB for tkinter
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Calculate scaling to fit canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:  # Ensure canvas is visible
            # Calculate scaling factor
            img_height, img_width = img_rgb.shape[:2]
            scale_w = canvas_width / img_width
            scale_h = canvas_height / img_height
            scale = min(scale_w, scale_h)
            
            # Resize image
            if scale < 1:
                new_width = int(img_width * scale)
                new_height = int(img_height * scale)
                img_rgb = cv2.resize(img_rgb, (new_width, new_height))
            
            # Convert to PIL Image
            pil_img = Image.fromarray(img_rgb)
            
            # Convert to PhotoImage
            self.tk_img = ImageTk.PhotoImage(image=pil_img)
            
            # Update canvas
            self.canvas.delete("all")
            x = (canvas_width - pil_img.width) // 2
            y = (canvas_height - pil_img.height) // 2
            self.canvas.create_image(x, y, image=self.tk_img, anchor=tk.NW)
    
    def on_resize(self, event):
        """Handle window resize event"""
        # Only process events from the main window
        if event.widget == self.root:
            # If there's a current frame, redisplay it to fit the new size
            if hasattr(self, 'current_frame') and self.current_frame is not None:
                # Add slight delay to ensure canvas has been resized
                self.root.after(100, lambda: self.display_image(self.current_frame))


def main():
    """Main function"""
    # Create the root window
    root = tk.Tk()
    
    # Set theme (if ttk theme extension is available)
    try:
        from ttkthemes import ThemedTk
        root = ThemedTk(theme="arc")
    except:
        pass
    
    # Create the application
    app = CarDetectionApp(root)
    
    # Start the main loop
    root.mainloop()


if __name__ == "__main__":
    main() 