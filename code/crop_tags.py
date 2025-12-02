import cv2
import numpy as np
import os
import json
from pathlib import Path
import matplotlib.pyplot as plt

class TagCropper:
    """Handle tag cropping with both auto and manual modes"""
    
    def __init__(self, bbox_file='bounding_boxes.json'):
        self.bbox_file = bbox_file
        self.bboxes = self._load_bboxes()
    
    def _load_bboxes(self):
        if Path(self.bbox_file).exists():
            with open(self.bbox_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_bboxes(self):
        with open(self.bbox_file, 'w') as f:
            json.dump(self.bboxes, f, indent=2)
    
    def auto_detect_tag(self, img, min_area=5000, padding=20):
        """Automatically detect tag using edge detection"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # More aggressive edge detection for better tag boundaries
        edges = cv2.Canny(blurred, 30, 100)
        
        # Dilate to connect nearby edges
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, 
                                        cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find the largest rectangular contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        if w * h < min_area:
            return None
        
        # Add padding
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img.shape[1] - x, w + 2*padding)
        h = min(img.shape[0] - y, h + 2*padding)
        
        return (x, y, w, h)
    
    def manual_crop_single(self, img_path):
        """Manual cropping with mouse interface"""
        img = cv2.imread(str(img_path))
        
        # Resize for display if too large
        display_img = img.copy()
        scale = 1.0
        max_display_height = 900
        if img.shape[0] > max_display_height:
            scale = max_display_height / img.shape[0]
            display_width = int(img.shape[1] * scale)
            display_img = cv2.resize(display_img, (display_width, max_display_height))
        
        img_display = display_img.copy()
        
        drawing = False
        ix, iy = -1, -1
        fx, fy = -1, -1
        
        def draw_rectangle(event, x, y, flags, param):
            nonlocal ix, iy, fx, fy, drawing, img_display
            
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                ix, iy = x, y
            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing:
                    img_display = display_img.copy()
                    cv2.rectangle(img_display, (ix, iy), (x, y), (0, 255, 0), 2)
            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                fx, fy = x, y
                cv2.rectangle(img_display, (ix, iy), (fx, fy), (0, 255, 0), 2)
        
        window_name = f'Crop: {Path(img_path).name} (s=save, r=reset, q=skip)'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, display_img.shape[1], display_img.shape[0])
        cv2.setMouseCallback(window_name, draw_rectangle)
        
        print(f"\nCropping: {Path(img_path).name}")
        print("  Draw box → 's' to save, 'r' to reset, 'q' to skip")
        
        while True:
            cv2.imshow(window_name, img_display)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):
                if ix != -1 and fx != -1:
                    # Scale coordinates back to original image size
                    x1 = int(min(ix, fx) / scale)
                    x2 = int(max(ix, fx) / scale)
                    y1 = int(min(iy, fy) / scale)
                    y2 = int(max(iy, fy) / scale)
                    
                    cv2.destroyAllWindows()
                    return (x1, y1, x2-x1, y2-y1)
            elif key == ord('r'):
                img_display = display_img.copy()
                ix, iy, fx, fy = -1, -1, -1, -1
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return None
        
        cv2.destroyAllWindows()
        return None
    
    def process_folder(self, input_folder, output_folder='cropped_tags', 
                       mode='hybrid'):
        """
        Process all images in folder
        mode: 'auto', 'manual', or 'hybrid'
        """
        os.makedirs(output_folder, exist_ok=True)
        
        # Get only .JPG files
        image_files = sorted(list(Path(input_folder).glob('*.JPG')))
        
        print(f"\n{'='*60}")
        print(f"CROPPING TAGS - Mode: {mode}")
        print(f"Found {len(image_files)} JPG images")
        print(f"{'='*60}")
        
        successful = 0
        skipped = 0
        failed = 0
        
        for i, img_path in enumerate(image_files):
            img_name = img_path.name
            
            # Skip if already processed
            if img_name in self.bboxes:
                print(f"[{i+1}/{len(image_files)}] ✓ Already cropped: {img_name}")
                successful += 1
                continue
            
            print(f"\n[{i+1}/{len(image_files)}] Processing: {img_name}")
            
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  ✗ Could not read image")
                failed += 1
                continue
            
            bbox = None
            
            # Try auto-detection
            if mode in ['auto', 'hybrid']:
                bbox = self.auto_detect_tag(img)
                
                if bbox:
                    print(f"  Auto-detected tag region")
            
            # Show preview and ask in hybrid mode
            if mode == 'hybrid' and bbox:
                x, y, w, h = bbox
                
                # Create preview
                img_with_box = img.copy()
                cv2.rectangle(img_with_box, (x, y), (x+w, y+h), (0, 255, 0), 5)
                
                # Resize for display
                max_height = 800
                if img_with_box.shape[0] > max_height:
                    scale = max_height / img_with_box.shape[0]
                    new_width = int(img_with_box.shape[1] * scale)
                    img_with_box_display = cv2.resize(img_with_box, (new_width, max_height))
                else:
                    img_with_box_display = img_with_box
                
                # Show with OpenCV (faster than matplotlib)
                cv2.imshow('Auto-detected crop (y/n/s)', img_with_box_display)
                print("  Accept auto-crop? (y=yes, n=manual, s=skip)")
                
                while True:
                    key = cv2.waitKey(0) & 0xFF
                    if key == ord('y'):
                        cv2.destroyAllWindows()
                        break
                    elif key == ord('n'):
                        cv2.destroyAllWindows()
                        bbox = None
                        break
                    elif key == ord('s'):
                        cv2.destroyAllWindows()
                        bbox = 'skip'
                        break
                
                if bbox == 'skip':
                    print(f"  Skipped by user")
                    skipped += 1
                    continue
            
            # Manual crop if needed
            if bbox is None and mode != 'auto':
                print(f"  Opening manual crop tool...")
                bbox = self.manual_crop_single(img_path)
                
                if bbox is None:
                    print(f"  Skipped by user")
                    skipped += 1
                    continue
            
            # Save cropped image
            if bbox:
                x, y, w, h = bbox
                cropped = img[y:y+h, x:x+w]
                output_path = Path(output_folder) / img_name
                cv2.imwrite(str(output_path), cropped)
                
                self.bboxes[img_name] = {'x': int(x), 'y': int(y), 
                                         'w': int(w), 'h': int(h)}
                self._save_bboxes()
                print(f"  ✓ Saved: {output_path}")
                successful += 1
            else:
                print(f"  ✗ No bounding box detected")
                failed += 1
        
        print(f"\n{'='*60}")
        print(f"CROPPING COMPLETE")
        print(f"{'='*60}")
        print(f"Successful: {successful}/{len(image_files)}")
        print(f"Skipped: {skipped}/{len(image_files)}")
        print(f"Failed: {failed}/{len(image_files)}")
        print(f"\nCropped images saved to: {output_folder}/")
        print(f"Bounding boxes saved to: {self.bbox_file}")
        
        return self.bboxes

# ============================================================================
# USAGE
# ============================================================================

if __name__ == "__main__":
    # Initialize cropper
    cropper = TagCropper(bbox_file='bounding_boxes.json')
    
    # Process all images
    # Options:
    # - 'auto': Fully automatic (fast, but may need manual fixes)
    # - 'manual': Manually crop each image (accurate, but slower)
    # - 'hybrid': Auto-detect + confirm each (recommended for 86 images)
    
    cropper.process_folder(
        input_folder='tag_images',
        output_folder='cropped_tags',
        mode='manual'  # Start with auto to see how it performs
    )