import json
import os
import cv2
import numpy as np

def create_mask_from_labelme_json(json_path, image_shape, output_path, label_to_fill=None):
    """
    Creates a binary mask image from a LabelMe JSON annotation file.

    It reads all polygon shapes from the JSON and fills them in a black image.
    If 'label_to_fill' is specified, only polygons with that specific label are filled.

    Args:
        json_path (str): Path to the LabelMe JSON file.
        image_shape (tuple): The shape of the original image (height, width). Used to create the mask canvas.
        output_path (str): Path to save the generated binary mask image (e.g., 'mask.png').
        label_to_fill (str, optional): If provided, only shapes with this label will be
                                       drawn on the mask. Otherwise, all shapes are drawn.
    """
    # --- 1. Load the JSON file ---
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON file {json_path}: {e}")
        return

    # --- 2. Create a blank (black) image with the same dimensions as the original image ---
    mask = np.zeros(image_shape[:2], dtype=np.uint8)

    # --- 3. Iterate through all shapes in the JSON file ---
    shapes_drawn = 0
    for shape in data['shapes']:
        # Check if we need to filter by a specific label
        if label_to_fill is not None and shape['label'] != label_to_fill:
            continue  # Skip this shape if its label doesn't match

        # Get the polygon points
        points = shape['points']
        
        # Convert points to a NumPy array of integer type
        polygon = np.array(points, dtype=np.int32)

        # --- 4. Draw the filled polygon on the mask ---
        # The color is white (255)
        cv2.fillPoly(mask, [polygon], 255)
        shapes_drawn += 1

    if shapes_drawn == 0:
        if label_to_fill:
            print(f"Warning: No shapes with label '{label_to_fill}' found in {json_path}. An empty mask will be saved.")
        else:
            print(f"Warning: No shapes found in {json_path}. An empty mask will be saved.")

    # --- 5. Save the generated mask ---
    try:
        # Get the directory of the output path and create it if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        cv2.imwrite(output_path, mask)
        print(f"Mask saved successfully to {output_path}")
    except Exception as e:
        print(f"Error saving mask image to {output_path}: {e}")


# --- Example Usage ---
if __name__ == '__main__':
    # --- Parameters ---
    # TODO: Replace these with your actual file paths
    # The original image is needed to get the correct height and width for the mask
    image_file = "/home/featurize/work/data/flower/coco_output/image_fly/1289_1080_1080.jpg"
    json_annotation_file = "/home/featurize/work/data/flower/flower1/1289_1080_1080.json"
    output_mask_file = "generated_mask_flower.png"
    
    # Optional: If your JSON has multiple objects and you only want to create a mask for one
    # specific object, specify its label here. If None, all objects will be in the mask.
    specific_label = None # e.g., "tussock grass" or None

    # --- Create dummy files for demonstration ---
    # Create a dummy image
    H, W = 480, 640
    dummy_img = np.zeros((H, W, 3), dtype=np.uint8)
    cv2.imwrite(image_file, dummy_img)
    
    # Create a dummy LabelMe JSON
    dummy_json_data = {
      "version": "5.0.1",
      "flags": {},
      "shapes": [
        {
          "label": "tussock grass",
          "points": [ [150.0, 100.0], [490.0, 100.0], [490.0, 400.0], [150.0, 400.0] ],
          "group_id": None,
          "shape_type": "polygon",
          "flags": {}
        },
        {
          "label": "rock",
          "points": [ [50.0, 50.0], [100.0, 50.0], [100.0, 150.0], [50.0, 150.0] ],
          "group_id": None,
          "shape_type": "polygon",
          "flags": {}
        }
      ],
      "imagePath": "dummy_image.jpg",
      "imageData": None,
      "imageHeight": H,
      "imageWidth": W
    }
    with open(json_annotation_file, 'w') as f:
        json.dump(dummy_json_data, f, indent=2)
    # --------------------------------------------------

    # --- Main Logic ---
    # First, get the shape of the original image
    try:
        img = cv2.imread(image_file)
        if img is None:
            raise FileNotFoundError(f"Image not found at {image_file}")
        img_shape = img.shape
    except Exception as e:
        print(f"Could not read image to get shape: {e}")
        # If image can't be read, try to get shape from JSON
        try:
            with open(json_annotation_file, 'r') as f:
                data = json.load(f)
            img_shape = (data['imageHeight'], data['imageWidth'])
            print("Got image shape from JSON file.")
        except Exception as json_e:
            print(f"Could not get image shape from JSON either: {json_e}")
            exit()

    # Generate the mask
    create_mask_from_labelme_json(
        json_path=json_annotation_file,
        image_shape=img_shape,
        output_path=output_mask_file,
        label_to_fill=specific_label
    )
    
    print("\n--- To generate a mask for a specific label ---")
    create_mask_from_labelme_json(
        json_path=json_annotation_file,
        image_shape=img_shape,
        output_path="generated_mask_specific_label.png",
        label_to_fill="tussock grass" # Now only this object will be in the mask
    )