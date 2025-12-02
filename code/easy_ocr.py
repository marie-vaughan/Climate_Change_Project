import easyocr
import cv2

# Load and resize image
img = cv2.imread('tag_images_cropped/IMG_8539 2.JPG')

# Resize to a more reasonable size (maintaining aspect ratio)
max_dimension = 1920
height, width = img.shape[:2]
if max(height, width) > max_dimension:
    scale = max_dimension / max(height, width)
    new_width = int(width * scale)
    new_height = int(height * scale)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    print(f"Resized to: {img.shape}")

reader = easyocr.Reader(['en'])
result = reader.readtext(img)

for (bbox, text, prob) in result:
    print(f'Text: {text}, Probability: {prob:.2f}')