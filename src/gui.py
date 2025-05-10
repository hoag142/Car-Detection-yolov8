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
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import threading
from pathlib import Path
from ttkthemes import ThemedTk
import tkinter.font as tkFont

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
        
        # Create menu
        self.create_menu()
    
    def create_ui(self):
        """Create the user interface"""
        # Thiết lập style
        self.setup_styles()
        
        # Thêm gradient header
        self.create_gradient_header(self.root, "Car Detection using YOLOv8")
        
        # Main layout với padding cải thiện
        self.main_frame = ttk.Frame(self.root, padding=15)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel (controls) với viền và bo góc
        self.left_panel = ttk.Frame(self.main_frame, padding=10, width=300, relief=tk.RIDGE, borderwidth=1)
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y, expand=False, padx=(0, 10))
        
        # Setup theme switcher
        self.theme_mode = tk.StringVar(value="light")
        self.create_theme_switcher()
        
        # Model section with improved styling
        model_frame = ttk.LabelFrame(self.left_panel, text="Model", padding=10)
        model_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(model_frame, text="Model Path:").pack(anchor=tk.W)
        model_path_frame = ttk.Frame(model_frame)
        model_path_frame.pack(fill=tk.X, pady=5)
        ttk.Entry(model_path_frame, textvariable=self.model_path).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(model_path_frame, text="Browse", command=self.browse_model).pack(side=tk.RIGHT)
        ttk.Button(model_frame, text="Load Model", command=self.load_model, style="Accent.TButton").pack(fill=tk.X)
        
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
        
        # Source section với nút hiện đại và icon
        source_frame = ttk.LabelFrame(self.left_panel, text="Source", padding=10)
        source_frame.pack(fill=tk.X, pady=5)
        
        # Đường dẫn tới thư mục assets
        assets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
        
        # Tạo thư mục assets nếu chưa tồn tại
        os.makedirs(assets_dir, exist_ok=True)
        
        # Tạo các nút với icon (nếu có)
        image_icon = os.path.join(assets_dir, "image_icon.png")
        video_icon = os.path.join(assets_dir, "video_icon.png")
        webcam_icon = os.path.join(assets_dir, "webcam_icon.png")
        stop_icon = os.path.join(assets_dir, "stop_icon.png")
        save_icon = os.path.join(assets_dir, "save_icon.png")
        
        # Nếu chưa có icon, sử dụng nút thông thường
        ttk.Button(source_frame, text="Open Image", command=self.open_image, style="Accent.TButton").pack(fill=tk.X, pady=2)
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
        self.show_loading_animation()
        
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

    def create_rounded_button(self, parent, text, command, icon_path=None):
        """Tạo nút với viền bo tròn và icon"""
        btn_frame = ttk.Frame(parent)
        
        # Tạo nút với style tùy chỉnh
        button = ttk.Button(btn_frame, text=text, command=command, style="Rounded.TButton")
        
        # Thêm icon nếu có
        if icon_path and os.path.exists(icon_path):
            try:
                icon = Image.open(icon_path).resize((20, 20))
                photo = ImageTk.PhotoImage(icon)
                button.image = photo  # Giữ tham chiếu
                button.configure(image=photo, compound=tk.LEFT, padding=(5, 5))
            except Exception as e:
                print(f"Could not load icon {icon_path}: {e}")
        
        button.pack(fill=tk.X, padx=5, pady=5)
        return btn_frame

    def create_gradient_header(self, parent, text, start_color="#007BFF", end_color="#00BFFF"):
        """Tạo header với gradient background"""
        header_frame = ttk.Frame(parent)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Canvas cho gradient
        canvas_height = 40
        canvas = tk.Canvas(header_frame, height=canvas_height, bd=0, highlightthickness=0)
        canvas.pack(fill=tk.X)
        
        # Vẽ gradient
        width = parent.winfo_screenwidth()  # Sử dụng chiều rộng màn hình
        for i in range(canvas_height):
            # Convert color từ HEX sang RGB
            r1, g1, b1 = parent.winfo_rgb(start_color)
            r2, g2, b2 = parent.winfo_rgb(end_color)
            # Tính toán màu gradient
            r = (r1 + int((r2-r1) * i/canvas_height)) // 256
            g = (g1 + int((g2-g1) * i/canvas_height)) // 256
            b = (b1 + int((b2-b1) * i/canvas_height)) // 256
            color = f'#{r:02x}{g:02x}{b:02x}'
            canvas.create_line(0, i, width, i, fill=color)
        
        # Thêm text vào header
        canvas.create_text(
            width // 2, canvas_height // 2, 
            text=text, 
            fill="white", 
            font=("Segoe UI", 14, "bold")
        )
        
        return header_frame

    def setup_styles(self):
        """Thiết lập styles cho UI elements"""
        style = ttk.Style()
        
        # Button styles
        style.configure("Accent.TButton", background="#007BFF", foreground="white")
        style.map("Accent.TButton",
                  background=[('active', '#0069D9'), ('disabled', '#6C757D')],
                  foreground=[('active', 'white'), ('disabled', '#A9A9A9')])
        
        # Frame styles
        style.configure("Card.TFrame", background="#FFFFFF", relief=tk.RAISED, borderwidth=1)
        
        # Label styles
        style.configure("Title.TLabel", font=("Segoe UI", 14, "bold"))
        style.configure("Subtitle.TLabel", font=("Segoe UI", 12))

    def create_theme_switcher(self):
        """Tạo công tắc chuyển đổi theme dark/light"""
        theme_frame = ttk.Frame(self.left_panel)
        theme_frame.pack(fill=tk.X, pady=5)
        
        theme_switch = ttk.Checkbutton(
            theme_frame, 
            text="Dark Mode", 
            variable=self.theme_mode,
            onvalue="dark", 
            offvalue="light",
            command=self.toggle_theme
        )
        theme_switch.pack(side=tk.RIGHT)

    def toggle_theme(self):
        """Chuyển đổi giữa dark mode và light mode"""
        if self.theme_mode.get() == "dark":
            # Áp dụng dark theme
            style = ttk.Style()
            style.theme_use("equilux")  # Sử dụng theme tối
            self.canvas.config(bg="#222222")
        else:
            # Áp dụng light theme
            style = ttk.Style()
            style.theme_use("arc")  # Sử dụng theme sáng
            self.canvas.config(bg="black")

    def show_loading_animation(self):
        """Hiển thị animation khi loading"""
        self.progress.start()
        self.status_var.set("Processing... Please wait")
        
        # Hiệu ứng loading
        loading_text = "Loading"
        for _ in range(3):
            for i in range(4):
                dots = "." * i
                self.status_var.set(f"{loading_text}{dots}")
                self.root.update()
                time.sleep(0.2)

    def create_menu(self):
        """Tạo thanh menu cho ứng dụng"""
        menu_bar = tk.Menu(self.root)
        
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Open Image", command=self.open_image)
        file_menu.add_command(label="Open Video", command=self.open_video)
        file_menu.add_command(label="Open Webcam", command=self.open_webcam)
        file_menu.add_separator()
        file_menu.add_command(label="Save Current Frame", command=self.save_frame)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menu_bar.add_cascade(label="File", menu=file_menu)
        
        view_menu = tk.Menu(menu_bar, tearoff=0)
        view_menu.add_radiobutton(label="Standard View", variable=self.visualization_mode, value="standard")
        view_menu.add_radiobutton(label="Blur Background", variable=self.visualization_mode, value="blur_bg")
        view_menu.add_radiobutton(label="Heatmap", variable=self.visualization_mode, value="heatmap")
        view_menu.add_radiobutton(label="Edge Enhancement", variable=self.visualization_mode, value="edge")
        menu_bar.add_cascade(label="View", menu=view_menu)
        
        help_menu = tk.Menu(menu_bar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        menu_bar.add_cascade(label="Help", menu=help_menu)
        
        self.root.config(menu=menu_bar)

    def show_about(self):
        """Hiển thị thông tin về ứng dụng"""
        messagebox.showinfo(
            "About Car Detection",
            "Car Detection using YOLOv8\n\n"
            "This application detects cars in images, videos, and webcam feeds "
            "using the YOLOv8 object detection model.\n\n"
            "© 2023 Digital Image Processing"
        )


def main():
    """Main function"""
    try:
        # Sử dụng ThemedTk thay vì Tk thông thường
        root = ThemedTk(theme="arc")  # Các theme khác: breeze, equilux, ubuntu, radiance
        root.title("Car Detection using YOLO")
        
        # Thiết lập style
        style = ttk.Style()
        style.configure("TButton", font=("Segoe UI", 10))
        style.configure("TLabel", font=("Segoe UI", 10))
        style.configure("TFrame", background=style.lookup("TFrame", "background"))
        
        # Tạo application
        app = CarDetectionApp(root)
        
        # Chạy main loop
        root.mainloop()
    except Exception as e:
        print(f"Error initializing GUI: {e}")
        # Fallback to standard Tk
        root = tk.Tk()
        app = CarDetectionApp(root)
        root.mainloop()


if __name__ == "__main__":
    main() 