#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Car Detection Application Launcher

This script serves as the entry point to the Car Detection application,
providing options to run either the GUI interface or the command-line
version with various options.
"""

import os
import sys
import argparse
from pathlib import Path


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Car Detection System using YOLOv8',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--gui', action='store_true',
                        help='Start the graphical user interface')
    
    parser.add_argument('--source', type=str, default=None,
                        help='Source for detection: path to image, video, or 0 for webcam')
    
    parser.add_argument('--model', type=str, default='models/best_car_detection.pt',
                        help='Path to trained YOLO model')
    
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold for detection')
    
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IoU threshold for Non-Maximum Suppression')
    
    parser.add_argument('--view', action='store_true',
                        help='Display results in a window')
    
    parser.add_argument('--save', action='store_true',
                        help='Save detection results')
    
    parser.add_argument('--output', type=str, default='outputs',
                        help='Output directory for saved results')
    
    parser.add_argument('--visualize', type=str, default='standard',
                        choices=['standard', 'blur_bg', 'heatmap', 'edge'],
                        help='Visualization mode')
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()
    
    # Ensure we are in the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    if args.gui:
        # Launch GUI application
        print("Starting Car Detection GUI...")
        from src.gui import main as gui_main
        gui_main()
    
    elif args.source is not None:
        # Run detection from command line
        print(f"Running car detection on {args.source}...")
        
        from src.detect import main as detect_main
        
        # Prepare arguments for detect.py
        sys.argv = [
            'detect.py',
            '--source', args.source,
            '--model', args.model,
            '--conf', str(args.conf),
            '--iou', str(args.iou),
            '--save-dir', args.output,
            '--visualize', args.visualize
        ]
        
        if args.view:
            sys.argv.append('--view-img')
            
        if args.save:
            sys.argv.append('--save-img')
            
        detect_main()
    
    else:
        # No arguments provided, show help message
        print("Car Detection System - YOLOv8")
        print("\nOptions:")
        print("  1. Run GUI: python launch.py --gui")
        print("  2. Process image: python launch.py --source path/to/image.jpg --view --save")
        print("  3. Process video: python launch.py --source path/to/video.mp4 --view --save")
        print("  4. Use webcam: python launch.py --source 0 --view")
        print("\nFor more options, run: python launch.py --help")


if __name__ == '__main__':
    main() 