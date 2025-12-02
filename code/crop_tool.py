import cv2
import os
import numpy as np
from pathlib import Path

def manual_crop_and_rotate(input_folder='tag_images', output_folder='cropped_tags'):
    """
    Manual cropping and rotation tool
    
    Controls:
        ROTATION (before cropping):
        - Press 'l' to rotate left (counter-clockwise) 90°
        - Press 'r' to rotate right (clockwise) 90°
        
        CROPPING:
        - Click and drag to draw box
        - Press 's' to save and move to next image
        - Press 'q' to skip this image
        - Press 'x' to reset (undo rotation and crop box)
        - Press ESC to quit entirely
    """
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all images
    image_files = sorted(list(Path(input_folder).glob('*.JPG')) + 
                        list(Path(input_folder).glob('*.jpg')) +
                        list(Path(input_folder).glob('*.png')))
    
    if not image_files:
        print(f"No images found in {input_folder}")
        return
    
    # Track which images already exist in output
    already_cropped = [f.name for f in Path(output_folder).glob('*')]
    
    print(f"\n{'='*60}")
    print(f"MANUAL CROP & ROTATE TOOL")
    print(f"{'='*60}")
    print(f"Images to process: {len(image_files)}")
    print(f"Already cropped: {len(already_cropped)}")
    print(f"Output folder: {output_folder}")
    print(f"\nControls:")
    print(f"  l = Rotate left (90°)")
    print(f"  r = Rotate right (90°)")
    print(f"  s = Save crop")
    print(f"  x = Reset (undo rotation & crop)")
    print(f"  q = Skip image")
    print(f"  ESC = Quit")
    print(f"{'='*60}\n")
    
    cropped_count = 0
    skipped_count = 0
    
    for i, img_path in enumerate(image_files):
        # Skip if already cropped
        if img_path.name in already_cropped:
            print(f"[{i+1}/{len(image_files)}] ✓ Already cropped: {img_path.name}")
            cropped_count += 1
            continue
        
        print(f"\n[{i+1}/{len(image_files)}] {img_path.name}")
        
        # Load image
        original_img = cv2.imread(str(img_path))
        
        if original_img is None:
            print(f"  Error: Could not load image")
            continue
        
        # Current working image (can be rotated)
        img = original_img.copy()
        rotation_count = 0  # Track rotations (0, 1, 2, 3 = 0°, 90°, 180°, 270°)
        
        # Prepare display
        def prepare_display(image):
            display_img = image.copy()
            scale = 1.0
            
            max_display_height = 900
            if image.shape[0] > max_display_height:
                scale = max_display_height / image.shape[0]
                new_width = int(image.shape[1] * scale)
                display_img = cv2.resize(display_img, (new_width, max_display_height))
            
            return display_img, scale
        
        display_img, scale = prepare_display(img)
        
        # Drawing state
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
                    cv2.rectangle(img_display, (ix, iy), (x, y), (0, 255, 0), 3)
                    
                    # Show rotation indicator
                    if rotation_count > 0:
                        cv2.putText(img_display, f"Rotated: {rotation_count * 90}°", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                                   (0, 255, 255), 2)
            
            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                fx, fy = x, y
                cv2.rectangle(img_display, (ix, iy), (fx, fy), (0, 255, 0), 3)
                
                # Show rotation indicator
                if rotation_count > 0:
                    cv2.putText(img_display, f"Rotated: {rotation_count * 90}°", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                               (0, 255, 255), 2)
        
        # Create window
        window_name = f'{img_path.name} - l/r=rotate, s=save, x=reset, q=skip, ESC=quit'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, draw_rectangle)
        
        # Add rotation indicator to initial display
        if rotation_count > 0:
            cv2.putText(img_display, f"Rotated: {rotation_count * 90}°", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                       (0, 255, 255), 2)
        
        # Main loop
        while True:
            cv2.imshow(window_name, img_display)
            key = cv2.waitKey(1) & 0xFF
            
            # Rotate left (counter-clockwise)
            if key == ord('l'):
                rotation_count = (rotation_count + 1) % 4
                img = np.rot90(original_img, rotation_count)
                display_img, scale = prepare_display(img)
                img_display = display_img.copy()
                ix, iy, fx, fy = -1, -1, -1, -1  # Reset crop box
                cv2.putText(img_display, f"Rotated: {rotation_count * 90}°", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                           (0, 255, 255), 2)
                print(f"  Rotated left → {rotation_count * 90}°")
            
            # Rotate right (clockwise)
            elif key == ord('r'):
                rotation_count = (rotation_count - 1) % 4
                img = np.rot90(original_img, rotation_count)
                display_img, scale = prepare_display(img)
                img_display = display_img.copy()
                ix, iy, fx, fy = -1, -1, -1, -1  # Reset crop box
                cv2.putText(img_display, f"Rotated: {rotation_count * 90}°", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                           (0, 255, 255), 2)
                print(f"  Rotated right → {rotation_count * 90}°")
            
            # Reset rotation and crop
            elif key == ord('x'):
                rotation_count = 0
                img = original_img.copy()
                display_img, scale = prepare_display(img)
                img_display = display_img.copy()
                ix, iy, fx, fy = -1, -1, -1, -1
                print(f"  Reset to original")
            
            # Save crop
            elif key == ord('s'):
                if ix != -1 and fx != -1:
                    # Scale coordinates back to original size
                    x1 = int(min(ix, fx) / scale)
                    x2 = int(max(ix, fx) / scale)
                    y1 = int(min(iy, fy) / scale)
                    y2 = int(max(iy, fy) / scale)
                    
                    # Crop at original resolution
                    cropped = img[y1:y2, x1:x2]
                    
                    # Save
                    output_path = Path(output_folder) / img_path.name
                    cv2.imwrite(str(output_path), cropped, 
                               [cv2.IMWRITE_JPEG_QUALITY, 100])
                    
                    rotation_msg = f" (rotated {rotation_count * 90}°)" if rotation_count > 0 else ""
                    print(f"  ✓ Saved: {output_path}{rotation_msg}")
                    cropped_count += 1
                    cv2.destroyAllWindows()
                    break
                else:
                    print("  No box drawn! Draw a box first.")
            
            # Skip
            elif key == ord('q'):
                print(f"  Skipped")
                skipped_count += 1
                cv2.destroyAllWindows()
                break
            
            # Quit entirely
            elif key == 27:  # ESC
                print(f"\n\nQuitting...")
                cv2.destroyAllWindows()
                print(f"\n{'='*60}")
                print(f"SUMMARY")
                print(f"{'='*60}")
                print(f"Cropped: {cropped_count}")
                print(f"Skipped: {skipped_count}")
                print(f"{'='*60}")
                return
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"COMPLETE!")
    print(f"{'='*60}")
    print(f"Total cropped: {cropped_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Output folder: {output_folder}")
    print(f"{'='*60}")

if __name__ == "__main__":
    # Run the tool
    manual_crop_and_rotate(
        input_folder='tag_images',
        output_folder='cropped_tags'
    )