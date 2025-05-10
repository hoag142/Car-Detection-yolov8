<<<<<<< HEAD
# Car-Detection-yolov8
DIP project
=======
# Car Detection using YOLO

This project implements car detection using YOLOv8, with data management through Roboflow and various image processing techniques.

[English](#english) | [Tiếng Việt](#vietnamese)

<a name="english"></a>
## English

### Project Structure
```
car_detection/
├── data/               # Data storage (training, validation, testing)
├── models/             # Trained model weights and configurations 
├── src/                # Source code for the application
│   ├── detect.py       # Script for detection on images and videos
│   └── gui.py          # GUI application
├── notebooks/          # Jupyter notebooks for experimentation
│   ├── data_preparation.ipynb  # Data preparation using Roboflow
│   └── model_training.ipynb    # Model training guide
├── utils/              # Utility functions
│   └── image_processing.py  # Image processing techniques
├── requirements.txt    # Project dependencies
├── launch.py           # Application launcher
└── README.md           # Project documentation
```

### Setup and Installation

1. Install project dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Setup data using Roboflow (instructions in notebooks/data_preparation.ipynb)

3. Train model (instructions in notebooks/model_training.ipynb)

4. Run application:
   ```
   # GUI mode
   python launch.py --gui
   
   # CLI mode (image)
   python launch.py --source path/to/image.jpg --view --save
   
   # CLI mode (video)
   python launch.py --source path/to/video.mp4 --view --save
   
   # CLI mode (webcam)
   python launch.py --source 0 --view
   ```

### Data Preparation with Roboflow

This project uses Roboflow for dataset management, augmentation, and preprocessing:

1. Create a free account on [Roboflow](https://roboflow.com/)
2. Create a new project or use existing car detection datasets
3. Apply preprocessing and augmentation:
   - Auto-orient images
   - Resize to 640x640
   - Apply image enhancements
   - Add augmentations: rotation, brightness adjustments, blur, etc.
4. Generate and download the dataset in YOLOv8 format
5. Refer to `notebooks/data_preparation.ipynb` for detailed instructions

### Image Processing Techniques Applied

The project incorporates various image processing techniques:

#### Pre-processing:
- Histogram equalization
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Color space transformations (RGB, HSV, LAB)
- Gaussian and bilateral filtering for noise reduction
- Image sharpening

#### Data augmentation:
- Rotation and flipping
- Brightness and contrast adjustments
- Color space variations
- Blur and noise addition
- Mosaic (combining multiple images)

#### Post-processing:
- Non-maximum suppression for overlapping detections
- Confidence thresholding
- Selective blur/focus effects
- Edge enhancement
- Heatmap visualization

### Training and Evaluation

Training instructions and evaluation metrics are provided in the `notebooks/model_training.ipynb` notebook:

1. Load the prepared dataset
2. Initialize YOLOv8 model with pretrained weights
3. Set training parameters
4. Train the model
5. Evaluate on validation set
6. Export for inference

### Inference and Visualization

The application offers multiple visualization modes:

1. **Standard**: Default YOLO visualization
2. **Blur Background**: Blurs the background, keeping only detected objects sharp
3. **Heatmap**: Visualizes detection confidence as a heatmap overlay
4. **Edge**: Enhances edges of detected objects

### Command-Line Options

```
usage: launch.py [-h] [--gui] [--source SOURCE] [--model MODEL] [--conf CONF] [--iou IOU]
                 [--view] [--save] [--output OUTPUT]
                 [--visualize {standard,blur_bg,heatmap,edge}]

Car Detection System using YOLOv8

options:
  -h, --help            show this help message and exit
  --gui                 Start the graphical user interface (default: False)
  --source SOURCE       Source for detection: path to image, video, or 0 for webcam (default: None)
  --model MODEL         Path to trained YOLO model (default: models/best_car_detection.pt)
  --conf CONF           Confidence threshold for detection (default: 0.25)
  --iou IOU             IoU threshold for Non-Maximum Suppression (default: 0.45)
  --view                Display results in a window (default: False)
  --save                Save detection results (default: False)
  --output OUTPUT       Output directory for saved results (default: outputs)
  --visualize {standard,blur_bg,heatmap,edge}
                        Visualization mode (default: standard)
```

<a name="vietnamese"></a>
## Tiếng Việt

### Cấu trúc dự án
```
car_detection/
├── data/               # Lưu trữ dữ liệu (đào tạo, xác thực, kiểm tra)
├── models/             # Trọng số và cấu hình của mô hình được đào tạo
├── src/                # Mã nguồn cho ứng dụng
│   ├── detect.py       # Script phát hiện trên hình ảnh và video
│   └── gui.py          # Ứng dụng giao diện đồ họa
├── notebooks/          # Sổ tay Jupyter cho thử nghiệm
│   ├── data_preparation.ipynb  # Chuẩn bị dữ liệu sử dụng Roboflow
│   └── model_training.ipynb    # Hướng dẫn đào tạo mô hình
├── utils/              # Các hàm tiện ích
│   └── image_processing.py  # Kỹ thuật xử lý hình ảnh
├── requirements.txt    # Các phụ thuộc của dự án
├── launch.py           # Bộ khởi chạy ứng dụng
└── README.md           # Tài liệu dự án
```

### Cài đặt và thiết lập

1. Cài đặt các thư viện phụ thuộc:
   ```
   pip install -r requirements.txt
   ```

2. Thiết lập dữ liệu sử dụng Roboflow (hướng dẫn trong notebooks/data_preparation.ipynb)

3. Đào tạo mô hình (hướng dẫn trong notebooks/model_training.ipynb)

4. Chạy ứng dụng:
   ```
   # Chế độ giao diện đồ họa
   python launch.py --gui
   
   # Chế độ dòng lệnh (hình ảnh)
   python launch.py --source đường/dẫn/đến/hình.jpg --view --save
   
   # Chế độ dòng lệnh (video)
   python launch.py --source đường/dẫn/đến/video.mp4 --view --save
   
   # Chế độ dòng lệnh (webcam)
   python launch.py --source 0 --view
   ```

### Chuẩn bị dữ liệu với Roboflow

Dự án này sử dụng Roboflow để quản lý, tăng cường và tiền xử lý dữ liệu:

1. Tạo tài khoản miễn phí trên [Roboflow](https://roboflow.com/)
2. Tạo dự án mới hoặc sử dụng bộ dữ liệu xe có sẵn
3. Áp dụng tiền xử lý và tăng cường:
   - Tự động định hướng hình ảnh
   - Thay đổi kích thước thành 640x640
   - Áp dụng các cải tiến hình ảnh
   - Thêm tăng cường: xoay, điều chỉnh độ sáng, làm mờ, v.v.
4. Tạo và tải xuống bộ dữ liệu ở định dạng YOLOv8
5. Tham khảo `notebooks/data_preparation.ipynb` để biết hướng dẫn chi tiết

### Các kỹ thuật xử lý hình ảnh được áp dụng

Dự án tích hợp nhiều kỹ thuật xử lý hình ảnh:

#### Tiền xử lý:
- Cân bằng histogram
- CLAHE (Cân bằng histogram thích ứng hạn chế độ tương phản)
- Chuyển đổi không gian màu (RGB, HSV, LAB)
- Lọc Gaussian và song phương để giảm nhiễu
- Làm sắc nét hình ảnh

#### Tăng cường dữ liệu:
- Xoay và lật
- Điều chỉnh độ sáng và độ tương phản
- Biến thể không gian màu
- Thêm làm mờ và nhiễu
- Mosaic (kết hợp nhiều hình ảnh)

#### Hậu xử lý:
- Triệt tiêu không cực đại cho các phát hiện chồng chéo
- Ngưỡng độ tin cậy
- Hiệu ứng làm mờ/tập trung có chọn lọc
- Tăng cường cạnh
- Hiển thị bản đồ nhiệt

### Đào tạo và đánh giá

Hướng dẫn đào tạo và các số liệu đánh giá được cung cấp trong notebook `notebooks/model_training.ipynb`:

1. Tải bộ dữ liệu đã chuẩn bị
2. Khởi tạo mô hình YOLOv8 với trọng số đã đào tạo trước
3. Thiết lập tham số đào tạo
4. Đào tạo mô hình
5. Đánh giá trên tập xác thực
6. Xuất mô hình để suy luận

### Suy luận và hiển thị

Ứng dụng cung cấp nhiều chế độ hiển thị:

1. **Standard**: Hiển thị YOLO mặc định
2. **Blur Background**: Làm mờ nền, chỉ giữ đối tượng được phát hiện sắc nét
3. **Heatmap**: Hiển thị độ tin cậy phát hiện dưới dạng lớp phủ bản đồ nhiệt
4. **Edge**: Tăng cường cạnh của các đối tượng được phát hiện

### Tùy chọn dòng lệnh

```
sử dụng: launch.py [-h] [--gui] [--source SOURCE] [--model MODEL] [--conf CONF] [--iou IOU]
                  [--view] [--save] [--output OUTPUT]
                  [--visualize {standard,blur_bg,heatmap,edge}]

Hệ thống phát hiện xe sử dụng YOLOv8

tùy chọn:
  -h, --help            hiển thị thông báo trợ giúp và thoát
  --gui                 Khởi động giao diện đồ họa (mặc định: False)
  --source SOURCE       Nguồn để phát hiện: đường dẫn đến hình ảnh, video, hoặc 0 cho webcam (mặc định: None)
  --model MODEL         Đường dẫn đến mô hình YOLO đã đào tạo (mặc định: models/best_car_detection.pt)
  --conf CONF           Ngưỡng độ tin cậy cho phát hiện (mặc định: 0.25)
  --iou IOU             Ngưỡng IoU cho Non-Maximum Suppression (mặc định: 0.45)
  --view                Hiển thị kết quả trong cửa sổ (mặc định: False)
  --save                Lưu kết quả phát hiện (mặc định: False)
  --output OUTPUT       Thư mục đầu ra cho kết quả đã lưu (mặc định: outputs)
  --visualize {standard,blur_bg,heatmap,edge}
                        Chế độ hiển thị (mặc định: standard)
``` 
>>>>>>> master
