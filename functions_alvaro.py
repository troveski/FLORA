import cv2
import numpy as np

def create_transparent_mask_for_classes(image_path, label_path, target_classes):
    """
    Reads an image and a YOLOv7 label file, and returns an image (as a numpy array)
    where the areas inside the bounding boxes for the specified target classes are transparent (alpha=0).

    YOLOv7 label format per line:
        <class> <x_center> <y_center> <width> <height>
    (coordinates normalized between 0 and 1)

    Parameters:
      image_path (str): Path to the input image.
      label_path (str): Path to the label file.
      target_classes (list): List of classes (as strings or integers) to process.
      
    Returns:
      image (numpy.ndarray): The modified image with transparent bounding boxes.
    """
    # Load the image preserving all channels
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Could not load image: " + image_path)
    
    # If the image does not have an alpha channel, convert it from BGR to BGRA
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    
    height, width = image.shape[:2]
    
    # Read the label file (YOLOv7 format)
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    # Convert target_classes to strings for uniform comparison
    target_classes_str = set(map(str, target_classes))
    
    for line in lines:
        line = line.strip()
        if not line:
            continue  # Skip empty lines
        
        parts = line.split()
        if len(parts) < 5:
            continue  # Skip invalid lines
        
        # First element is the class label
        class_label = parts[0]
        if class_label not in target_classes_str:
            continue  # Skip if this class is not in the target list
        
        # Parse YOLOv7 bounding box values (normalized)
        x_center_norm = float(parts[1])
        y_center_norm = float(parts[2])
        bbox_w_norm   = float(parts[3])
        bbox_h_norm   = float(parts[4])
        
        # Convert normalized values to pixel coordinates
        x_center = x_center_norm * width
        y_center = y_center_norm * height
        bbox_w   = bbox_w_norm * width
        bbox_h   = bbox_h_norm * height
        
        # Calculate top-left and bottom-right pixel coordinates
        x1 = int(x_center - bbox_w / 2)
        y1 = int(y_center - bbox_h / 2)
        x2 = int(x_center + bbox_w / 2)
        y2 = int(y_center + bbox_h / 2)
        
        # Clamp coordinates to the image boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)
        
        # Set the alpha channel (fourth channel) to 0 for the region inside the bounding box
        image[y1:y2, x1:x2, 3] = 0
    
    # Return the modified image as a numpy array
    return image

# Example usage:
# result_img = create_transparent_mask_for_classes('input_image.png', 'label.txt', target_classes=[0, 2])
# Now, result_img holds the image with the transparent regions.

