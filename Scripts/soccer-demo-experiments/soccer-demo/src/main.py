import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from tqdm import tqdm

# Homography functions --------------------------------------------------------
def apply_homography_to_point(H, point):
    # Convert point to homogeneous coordinates
    point_homogeneous = np.append(point, 1)
    
    # Apply the homography matrix
    point_transformed_homogeneous = np.dot(H, point_homogeneous)
    
    # Convert back to Cartesian coordinates
    point_transformed = point_transformed_homogeneous[:2] / point_transformed_homogeneous[2]
    
    return point_transformed

# Color segmentation code -----------------------------------------------------
def save_hsv_ranges(hsv_ranges, file_path):
    # Convert numpy arrays to lists for JSON serialization
    hsv_ranges_list = [(lower.tolist(), upper.tolist()) for lower, upper in hsv_ranges]
    with open(file_path, 'w') as file:
        json.dump(hsv_ranges_list, file)

def load_hsv_ranges(file_path):
    with open(file_path, 'r') as file:
        hsv_ranges_list = json.load(file)
    # Convert lists back to numpy arrays
    hsv_ranges = [(np.array(lower), np.array(upper)) for lower, upper in hsv_ranges_list]
    return hsv_ranges

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None, None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame_index = total_frames // 2

    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        return None, None

    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)
    ret, middle_frame = cap.read()
    if not ret:
        print("Error: Could not read the middle frame.")
        return None, None

    cap.release()
    return first_frame, middle_frame

def scale_and_concat(frames):
    scaled_frames = [cv2.resize(frame, (frame.shape[1] // 3, frame.shape[0] // 3)) for frame in frames]
    concatenated_frame = np.concatenate(scaled_frames, axis=1)
    return concatenated_frame

def setup_hsv_ranges(config, n_classes=2):
    if config['hsv_ranges_path'] and os.path.exists(config['hsv_ranges_path']):
        # Load HSV ranges from the specified file
        return load_hsv_ranges(config['hsv_ranges_path'])
    
    else:
        # Proceed with manual setup of HSV ranges
        first_frame, middle_frame = extract_frames(config['input_video_path'])
        if first_frame is None or middle_frame is None:
            return
    
        concatenated_frame = scale_and_concat([first_frame, middle_frame])
        segmentation = VideoSegmentation('Segmentation')
        hsv_ranges_or_nans = []
        
        for _ in range(n_classes):
            while True:
                segmentation.update_segmentation(concatenated_frame)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('y'):
                    hsv_ranges_or_nans.append(segmentation.get_hsv_ranges())
                    break
                elif key == ord('n'):
                    hsv_ranges_or_nans.append(None)
                    break
                
            segmentation.reset_trackbars()
    
        cv2.destroyAllWindows()
        
        # Save the manually setup HSV ranges to a file in the 'outputs' directory
        save_hsv_ranges(hsv_ranges_or_nans, config['hsv_ranges_path'])
        
        return hsv_ranges_or_nans

class VideoSegmentation:
    def __init__(self, window_name):
        self.window_name = window_name
        cv2.namedWindow(self.window_name)
        self.lower_bound = np.array([0, 0, 0])
        self.upper_bound = np.array([179, 255, 255])
        self.create_trackbars()

    def create_trackbars(self):
        """Creates trackbars for HSV range selection."""
        cv2.createTrackbar('H Min', self.window_name, 0, 179, self.noop)
        cv2.createTrackbar('H Max', self.window_name, 179, 179, self.noop)
        cv2.createTrackbar('S Min', self.window_name, 0, 255, self.noop)
        cv2.createTrackbar('S Max', self.window_name, 255, 255, self.noop)
        cv2.createTrackbar('V Min', self.window_name, 0, 255, self.noop)
        cv2.createTrackbar('V Max', self.window_name, 255, 255, self.noop)

    def noop(self, x):
        """No-operation function for trackbar callback."""
        pass

    def get_hsv_ranges(self):
        """Returns the current HSV range selections."""
        
        return self.lower_bound, self.upper_bound
    
    def update_segmentation(self, concatenated_frame):
        """Updates the segmentation based on trackbar positions."""
        hsv = cv2.cvtColor(concatenated_frame, cv2.COLOR_BGR2HSV)
        self.lower_bound = np.array([cv2.getTrackbarPos('H Min', self.window_name), 
                                     cv2.getTrackbarPos('S Min', self.window_name), 
                                     cv2.getTrackbarPos('V Min', self.window_name)])
        self.upper_bound = np.array([cv2.getTrackbarPos('H Max', self.window_name), 
                                     cv2.getTrackbarPos('S Max', self.window_name), 
                                     cv2.getTrackbarPos('V Max', self.window_name)])
        mask = cv2.inRange(hsv, self.lower_bound, self.upper_bound)        
        segmented = cv2.bitwise_and(concatenated_frame, concatenated_frame, mask=mask)
        cv2.imshow(self.window_name, segmented)
        return self.lower_bound, self.upper_bound
    
    def reset_trackbars(self):
        """Resets all trackbars to their initial values."""
        cv2.setTrackbarPos('H Min', self.window_name, 0)
        cv2.setTrackbarPos('H Max', self.window_name, 179)
        cv2.setTrackbarPos('S Min', self.window_name, 0)
        cv2.setTrackbarPos('S Max', self.window_name, 255)
        cv2.setTrackbarPos('V Min', self.window_name, 0)
        cv2.setTrackbarPos('V Max', self.window_name, 255)


# Utility functions -----------------------------------------------------------
def create_output_dirs(config):
    # Extract the video name from the input video path
    video_name = os.path.splitext(os.path.basename(config['input_video_path']))[0]
    
    # Create the path for the output video directory
    output_video_dir = os.path.join(config['output_base_dir'], video_name)
    config['output_video_dir'] = output_video_dir
    
    # Check if the directory exists, if not create it
    if not os.path.exists(output_video_dir):
        os.makedirs(output_video_dir)
    
    # Set the path for the output CSV file
    config['output_csv_path'] = os.path.join(output_video_dir, 'video_detections.csv')
    
    config['output_heatmap_1_image_path'] = os.path.join(output_video_dir, 'overlay_heatmap_team_1.png')
    config['output_heatmap_2_image_path'] = os.path.join(output_video_dir, 'overlay_heatmap_team_2.png')
    
    config['output_composite_video_path'] = os.path.join(output_video_dir, 'composite_video.mp4')
    
    # Set the path for the HSV ranges text file
    config['hsv_ranges_path'] = os.path.join(output_video_dir, 'hsv_ranges.txt')
    
    # Set the path for the H matrix numpy
    config['h_matrix_path'] = os.path.join(output_video_dir, 'h_matrix.npy')
    
    return config

# Homography matrix computation -----------------------------------------------
def load_and_prepare_images(layout_path, video_path):
    # Load images
    layout_img = cv2.imread(layout_path)
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    # Ensure the frame is read successfully
    if not ret:
        print("Error: Could not read the video frame.")
        return None, None

    return layout_img, frame


def compute_homography_matrix(config, layout_img, first_frame):
    """
    Computes the homography matrix based on user-selected points
    from the layout image and the first video frame.
    """
    # Check if the homography matrix file exists and load it if it does
    h_matrix_path = config['h_matrix_path']
    if h_matrix_path and os.path.exists(h_matrix_path):
        H = np.load(h_matrix_path)
        return H
    
    # Ensuring both images are of the same size by adding black padding to the smaller image
    max_height = max(layout_img.shape[0], first_frame.shape[0])
    max_width = max(layout_img.shape[1], first_frame.shape[1])

    padded_layout_img = cv2.copyMakeBorder(layout_img, 0, max_height - layout_img.shape[0], 0, max_width - layout_img.shape[1], cv2.BORDER_CONSTANT, value=[0, 0, 0])
    padded_first_frame = cv2.copyMakeBorder(first_frame, 0, max_height - first_frame.shape[0], 0, max_width - first_frame.shape[1], cv2.BORDER_CONSTANT, value=[0, 0, 0])

    # Concatenating both images horizontally
    concatenated_img = np.concatenate((padded_layout_img, padded_first_frame), axis=1)

    # Display the concatenated image and collect points
    cv2.namedWindow("Homography Points Selection", cv2.WINDOW_NORMAL)
    cv2.imshow("Homography Points Selection", concatenated_img)

    points_layout = []
    points_frame = []

    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            if x < max_width:  # Clicked on the layout image
                points_layout.append((x, y))
                cv2.circle(concatenated_img, (x, y), 5, (255, 0, 0), -1)
            else:  # Clicked on the video frame
                points_frame.append((x - max_width, y))
                cv2.circle(concatenated_img, (x, y), 5, (0, 255, 0), -1)

            if len(points_layout) == len(points_frame) and len(points_layout) > 0:
                for i in range(len(points_layout)):
                    cv2.line(concatenated_img, points_layout[i], (points_frame[i][0] + max_width, points_frame[i][1]), (0, 0, 255), 2)

            cv2.imshow("Homography Points Selection", concatenated_img)

    cv2.setMouseCallback("Homography Points Selection", click_event)

    print("Click corresponding points on the layout image and the video frame. Press 'q' to quit and calculate homography.")
    while True:
        cv2.imshow("Homography Points Selection", concatenated_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    
    if len(points_layout) >= 4 and len(points_frame) >= 4:
        H, _ = cv2.findHomography(np.array(points_frame), np.array(points_layout))
        # Save the homography matrix if the path is specified
        if h_matrix_path:
            print(H)
            np.save(h_matrix_path, H)
        return H
    else:
        print("Not enough points to compute homography.")
        return None
    
def apply_homography_and_display(layout_img, frame, H):
    """
    Applies the homography matrix to the input frame and displays it
    next to the final visualization of the concatenated image with points and lines.

    :param layout_img: The layout image.
    :param frame: The first frame of the video.
    :param H: The homography matrix.
    :param final_visualization: The final visualization image with points and lines.
    """
    # Warp the video frame to the layout image's perspective
    height, width, _ = layout_img.shape
    warped_frame = cv2.warpPerspective(frame, H, (width, height))
    
    plt.imshow(warped_frame)
    plt.show()
    
# Object detection functions --------------------------------------------------
class DetectedObject:
    def __init__(self, bbox, obj_id, lbl):
        self.bbox = bbox
        self.id = obj_id
        self.label = lbl
        self.point_2d = None
    def get_bbox_bottom(self):
        xmin, ymin, xmax, ymax = self.bbox
        return np.array([int((xmin + xmax) / 2), int(ymax)])
        
class DetectionProcessor:
    def __init__(self, model_path, classes_hsv_ranges):
        self.model = YOLO(model_path)
        self.classes_hsv_ranges = classes_hsv_ranges
        self.n_classes = len(self.classes_hsv_ranges)

    def compute_detected_objects(self, yolo_detections, frame):
        detected_objects = []
        detections_as_xyxy = yolo_detections[0].boxes.xyxy
        labels = yolo_detections[0].boxes.cls
        
        # for det_xyxy, det_id in zip(detections_as_xyxy, detections_ids):
        for lbl, det_xyxy in zip(labels, detections_as_xyxy):
            det_xyxy = det_xyxy.cpu().numpy()
            lbl = int(lbl.cpu().numpy())

            x1, y1, x2, y2 = det_xyxy
            frame_crop = frame[int(y1):int(y2)+1, int(x1):int(x2)+1, :]
            det_id = self.predict_class_by_color(frame_crop)
            
            det_object = DetectedObject(det_xyxy, det_id, lbl)
            detected_objects.append(det_object)
        
        return detected_objects
    
    def predict_class_by_color(self, frame_crop):
        frame_crop_hsv = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2HSV)
        class_scores = np.zeros((self.n_classes,), dtype=np.float32)

        for i, hsv_ranges in enumerate(self.classes_hsv_ranges):
            lower_bound, upper_bound = hsv_ranges
            mask = cv2.inRange(frame_crop_hsv, lower_bound, upper_bound)

            class_scores[i] = np.sum(mask)

        return np.argmax(class_scores) + 1


    def id_to_color(self, box_id):
        # Example strategy: cycle through a list of predefined BGR colors
        colors = [
            (0, 255, 0),  # Green
            (255, 0, 0),  # Blue
            (0, 0, 255),  # Red
            (0, 255, 255), # Cyan
            (255, 0, 255), # Magenta
            (255, 255, 0)  # Yellow
        ]
        # Use box_id to select a color, ensuring it cycles through the list
        return colors[box_id % len(colors)]    

    def draw_detected_objects(self, original_frame, detected_objects):
        frame = original_frame.copy()
        
        # Fixed dimensions for the triangle
        tri_height = 25
        tri_base_half_length = 15
        vertical_offset = 20
        
        for detected_obj in detected_objects:
            x1, y1, x2, y2 = detected_obj.bbox
            
            x1, x2 = sorted((x1, x2))
            y1, y2 = sorted((y1, y2))
            
            width, height = (x2 - x1), (y2 - y1)
            center = (x1 + width/2, y1 + height/2)
            
            # Draw only detections of "ball" label
            if detected_obj.label == 0:
                # Choose the smaller dimension of width/height for the radius to ensure the circle fits inside the bbox
                radius = int(min(width, height) / 2)
                
                center = [int(center[0]), int(center[1])]
                # Draw simple circle using the obtained center and width
                cv2.circle(frame, center, radius, (0, 255, 255), 2)  # Yellow circle with a thickness of 2
            
            # Draw only detections of "player" label
            else:
                box_id = detected_obj.id
                
                circle_color = self.id_to_color(box_id)
                
                # Calculate the bottom center of the bounding box for the ellipse
                bottom_center = (int(center[0]), int(center[1] + height / 2))
                
                # Define the axes for the ellipse and angle
                ellipse_axes = (int(width / 2), int(height / 10))
                ellipse_angle = 0
                ellipse_thickness = 4
    
                # Draw the bottom half ellipse in blue with the specified thickness
                cv2.ellipse(frame, bottom_center, ellipse_axes, ellipse_angle, 0, 180, circle_color, ellipse_thickness)
    
                # Calculate the bottom point of the triangle (above the bounding box)
                top_point_triangle = (int(center[0]), int(center[1] - height / 2) - vertical_offset)
    
                # Triangle points
                p1 = (top_point_triangle[0], top_point_triangle[1] + tri_height)
                p2 = (top_point_triangle[0] - tri_base_half_length, top_point_triangle[1])
                p3 = (top_point_triangle[0] + tri_base_half_length, top_point_triangle[1])
    
                # Draw the filled triangle in white for the ID
                cv2.drawContours(frame, [np.array([p1, p2, p3])], 0, (255, 255, 255), -1)
    
                # Add the ID text in black, centered in the triangle
                text_size = cv2.getTextSize(str(box_id), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                text_x = p1[0] - text_size[0] // 2
                text_y = p1[1] - 2* text_size[1] // 3
                cv2.putText(frame, str(box_id), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        return frame
    
    def draw_transformed_points_with_heatmap(self, original_frame, detected_objects, heatmap):
        frame = original_frame.copy()
        
        for detected_obj in detected_objects:
            # Center coordinates of the circle
            x, y = detected_obj.point_2d
            x, y = int(x), int(y)
            # Circle color in BGR format
            circle_color = self.id_to_color(detected_obj.id)
            
            # Draw a circle with a black border
            border_thickness = 3  # Thickness of the border
            circle_radius = 10    # Radius of the circle
            border_color = (0, 0, 0)  # Black color in BGR format
            
            id_index = detected_obj.id - 1  # Adjust id to be 0-based index
        
            # Create a mask for the circle
            mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
            cv2.circle(mask, (x, y), circle_radius, (1,), thickness=-1)  # Draw circle with value 1 on the mask
    
            # Update the specific heatmap channel based on detected_obj.id using the mask
            heatmap[:, :, id_index] += mask
            
            # First, draw the border circle
            cv2.circle(frame, (x, y), circle_radius + border_thickness, border_color, thickness=-1)
            # Then, draw the color fill circle
            cv2.circle(frame, (x, y), circle_radius, circle_color, thickness=-1)
            
        return frame, heatmap
    
    
# Drawing and visualization functions -----------------------------------------
def visualize_separate_heatmaps(heatmap, base=10):
    num_channels = heatmap.shape[2]
    heatmaps_colored = []
    
    for i in range(num_channels):
        # Apply logarithmic scaling to each channel
        heatmap_log = np.log1p(heatmap[:, :, i]) / np.log(base)
        
        # Normalize the heatmap for display
        heatmap_normalized = cv2.normalize(heatmap_log, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_uint8 = np.uint8(heatmap_normalized)
        
        # Apply a colormap for visualization
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmaps_colored.append(heatmap_colored)
    
    return heatmaps_colored

def overlay_heatmap_on_image(image, heatmap_colored, alpha=0.5):
    # Convert the original image to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Convert the grayscale image back to BGR to match the heatmap's channels
    grayscale_image_bgr = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2BGR)
    
    # Overlay the heatmap on the converted image
    # You might adjust the alpha value here or the weights to prioritize the heatmap visibility
    overlaid_image = cv2.addWeighted(grayscale_image_bgr, 1 - alpha, heatmap_colored, alpha, 0)
    
    return overlaid_image

# General video processing functions ------------------------------------------

class VideoProcessor:
    def __init__(self, config, classes_hsv_ranges, H=None):
        self.config = config
        self.detection_processor = DetectionProcessor(config["yolo_model_path"], classes_hsv_ranges)
        self.H = H
        self.template_img = cv2.imread(config['input_layout_image'], cv2.IMREAD_COLOR)
        self.key_points_layout = np.load(config['input_layout_array'])
        self.heatmap = np.zeros((self.template_img.shape[0], self.template_img.shape[1], 2), dtype=np.float32)
        self.heatmap_overlay_1 = np.zeros_like(self.template_img)
        self.heatmap_overlay_2 = np.zeros_like(self.template_img)
        
    def process_frame(self, frame):
        detections = self.detection_processor.model.track(frame, persist=True, verbose=False, tracker="botsort.yaml", conf=0.5, imgsz=640)
        detected_objects = self.detection_processor.compute_detected_objects(detections, frame)
        frame_with_detections = self.detection_processor.draw_detected_objects(frame, detected_objects)
        
        H = self.H
        
        if H is not None:
            for i, det_obj in enumerate(detected_objects):
                detected_objects[i].point_2d = apply_homography_to_point(H, det_obj.get_bbox_bottom())
                
            template_with_detections, self.heatmap = self.detection_processor.draw_transformed_points_with_heatmap(self.template_img, 
                                                                                                                   detected_objects, 
                                                                                                                   self.heatmap)
        else:
            template_with_detections = self.template_img
        
        
        # TODO
        #   - Find a way to implement filtering --> Possibly by measuring the points scattering level
        #   - Implement player re-identification
        
        return frame_with_detections, template_with_detections, self.heatmap
        
    # Initial function provided by the user
    def process_video(self):
        cap = cv2.VideoCapture(self.config['input_video_path'])
    
        # Check if video opened successfully
        if not cap.isOpened():
            print("Error opening video stream or file")
            return
    
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        layout_width = self.template_img.shape[1] // 2  # New size calculation
        layout_height = self.template_img.shape[0] // 2  # New size calculation
        
        fps = cap.get(cv2.CAP_PROP_FPS)
    
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID' if 'mp4v' does not work
        
        # Adjusted output size for single composite video
        out_composite = cv2.VideoWriter(self.config['output_composite_video_path'], fourcc, fps, (layout_width * 2, layout_height * 2))
        
        with tqdm(total=total_frames, desc="Processing video frames") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    processed_frame, layout_2d, heatmap = self.process_frame(frame)
                    
                    # Generate heatmaps overlay for both teams
                    colored_heatmaps = visualize_separate_heatmaps(heatmap)
                    
                    heatmap_overlay_1 = overlay_heatmap_on_image(self.template_img, colored_heatmaps[0], alpha=0.7)
                    heatmap_overlay_2 = overlay_heatmap_on_image(self.template_img, colored_heatmaps[1], alpha=0.7)

                    self.heatmap_overlay_1 = heatmap_overlay_1
                    self.heatmap_overlay_2 = heatmap_overlay_2
    
                    # Resize images
                    small_frame = cv2.resize(processed_frame, (layout_width, layout_height))
                    small_layout = cv2.resize(layout_2d, (layout_width, layout_height))
                    small_heatmap_1 = cv2.resize(heatmap_overlay_1, (layout_width, layout_height))
                    small_heatmap_2 = cv2.resize(heatmap_overlay_2, (layout_width, layout_height))
    
                    # Create a blank canvas
                    canvas = np.zeros((layout_height * 2, layout_width * 2, 3), dtype=np.uint8)
    
                    # Place images on the canvas
                    canvas[0:layout_height, 0:layout_width] = small_frame
                    canvas[0:layout_height, layout_width:layout_width*2] = small_layout
                    canvas[layout_height:layout_height*2, 0:layout_width] = small_heatmap_1
                    canvas[layout_height:layout_height*2, layout_width:layout_width*2] = small_heatmap_2
    
                    # Write the composite frame into the file 'output_composite_video_path'
                    out_composite.write(canvas)
    
                    # Display the single composite window
                    cv2.imshow("Composite", canvas)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
    
                    pbar.update(1)
                else:
                    break
    
        # Save images
        cv2.imwrite(config['output_heatmap_1_image_path'], self.heatmap_overlay_1)
        cv2.imwrite(config['output_heatmap_2_image_path'], self.heatmap_overlay_2)

	# Release everything if job is finished
        cap.release()
        out_composite.release()
        cv2.destroyAllWindows()

# Main function ---------------------------------------------------------------
if __name__ == "__main__":
    config = {
        'input_video_path': '../../../Datasets/demo/demo_v3.mp4',
        'input_layout_image': '../../../Datasets/soccer field layout/soccer_field_layout.png',
        'input_layout_array': '../../../Datasets/soccer field layout/soccer_field_layout_points.npy',
        'yolo_model_path': '../../../Models/pretrained-model/pretrained-yolov8-soccer.pt',
        'output_base_dir': '../outputs'
    }
    
    config = create_output_dirs(config)
    classes_hsv_ranges = setup_hsv_ranges(config, n_classes=2)
    layout_img, first_frame = load_and_prepare_images(config['input_layout_image'], config['input_video_path'])
    H = compute_homography_matrix(config, layout_img, first_frame)
    if H is not None:
        # apply_homography_and_display(layout_img, first_frame, H)
        processor = VideoProcessor(config, classes_hsv_ranges, H)
        processor.process_video()
    else:
        print("Homography matrix couldn't be loaded...")
    
    
    
    
    
    