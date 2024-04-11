import os
import json
import numpy as np
import cv2
from ultralytics import YOLO
import random
import math
import sys
from tqdm import tqdm
import string
import csv
import argparse

import time
import socket
import struct
import pickle
import asyncio
import sqlite3
import websockets
import base64
import datetime

# Utility functions ===========================================================
def create_output_dirs(config):
    # Extract the video name from the input video path
    video_name = os.path.splitext(os.path.basename(str(config['input_video_path'])))[0]
    
    # Create the path for the output video directory
    output_video_dir = os.path.join(config['output_base_dir'], video_name)
    config['output_video_dir'] = output_video_dir

    # Check if the directory exists, if not create it
    if not os.path.exists(output_video_dir):
        os.makedirs(output_video_dir)
    
    # Define paths for input data
    config['hsv_ranges_path'] = os.path.join(output_video_dir, 'hsv_ranges.json')
    config['h_matrix_path'] = os.path.join(output_video_dir, 'h_matrix.npy')
    config['goal_polygon_path'] = os.path.join(output_video_dir, 'goal_polygon.npy')
    
    # Define paths for output data
    config['output_video_path'] = os.path.join(output_video_dir, 'scores_video.mp4')
    config['layout_video_path'] = os.path.join(output_video_dir, 'layout_video.mp4')
    config['output_csv'] = os.path.join(output_video_dir, 'soccer_analytics.csv')
    config['output_db'] = os.path.join(output_video_dir, 'soccer_analytics.sqlite')
    config['output_report'] = os.path.join(output_video_dir, 'soccer_analytics_report.json')
    
    
    return config

def create_heatmaps_dirs(config, teams_dict):
    heatmap_video_path_dict = {}
    heatmap_image_path_dict = {}
    
    for key in teams_dict.keys():
        team_letter = teams_dict[key]['team_letter']
        
        heatmap_video_path = os.path.join(config['output_video_dir'], f'heatmap_video_{team_letter}.mp4')
        heatmap_image_path = os.path.join(config['output_video_dir'], f'heatmap_image_{team_letter}.png')
        
        heatmap_video_path_dict[team_letter] = heatmap_video_path
        heatmap_image_path_dict[team_letter] = heatmap_image_path
        
    config['output_video_heatmaps'] = heatmap_video_path_dict
    config['output_image_heatmaps'] = heatmap_image_path_dict
        
    return config


class DatabaseWriter:
    def __init__(self, config):
        """
        Initializes the DatabaseWriter object, setting up the database connection.

        :param config: Configuration dictionary containing 'output_db' as the path to the SQLite database.
        """
        self.db_path = config['output_db']  # Set the path for the SQLite database
        self.connect_db()  # Connect to the SQLite database

    def connect_db(self):
        """
        Connects to the SQLite database, creating it if it doesn't exist.
        """ 
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        # Check if the 'analytics' table exists
        self.cursor = self.conn.cursor()
        
        # Drop the existing 'analytics' table if it exists
        self.cursor.execute("DROP TABLE IF EXISTS analytics")
        
        # Create a new 'analytics' table with the desired schema
        self.cursor.execute('''CREATE TABLE analytics (
                                frame INTEGER,
                                a_x REAL,
                                a_y REAL,
                                b_x REAL,
                                b_y REAL,
                                ball_x REAL,
                                ball_y REAL,
                                ball_possession TEXT,
                                goals_team_a INTEGER,
                                goals_team_b INTEGER,
                                timestamp TEXT)''')
        
        self.conn.commit()

    def update_db(self, analytics_dict, frame_number_p, scores_dict):
        """
        Updates the SQLite database by inserting a new row with the latest data.
        """
        # Check and convert only if the object is a NumPy array
        def convert_to_list(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj  # Return the object as is if it's not a NumPy array

        # Extract data for each column, ensuring any NumPy arrays are converted to lists
        current_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        a_coords = analytics_dict.get('a', [(None, None)])[0]
        b_coords = analytics_dict.get('b', [(None, None)])[0]
        ball_coords = analytics_dict.get('ball', [(None, None)])[0]
        
        a_x, a_y = a_coords if a_coords is not None else (None, None)
        b_x, b_y = b_coords if b_coords is not None else (None, None)
        ball_x, ball_y = ball_coords if ball_coords is not None else (None, None)
        
        ball_possession_str = json.dumps(analytics_dict.get('ball_possession', []))

        # Assume frame_number is calculated or obtained elsewhere
        frame_number = frame_number_p  # Example to derive frame_number, adjust as necessary
        # Get the goal counts for each team from the scores_dict
        goals_team_a = scores_dict.get('a', 0)
        goals_team_b = scores_dict.get('b', 0)
        # Insert the data into the database
        self.cursor.execute('''INSERT INTO analytics (frame, a_x, a_y, b_x, b_y, ball_x, ball_y, ball_possession, goals_team_a, goals_team_b, timestamp)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                            (frame_number, a_x, a_y, b_x, b_y, ball_x, ball_y, ball_possession_str, goals_team_a, goals_team_b, current_timestamp))
        self.conn.commit()

    def close_db(self):
        """
        Closes the database connection.
        """
        self.conn.close()
           
# TXT writer for final analytics report =======================================
class ReportWriter:
    def __init__(self, config):
        self.file_path = config['output_report']
        # Ensure the CSV starts fresh when the object is created
        self.ensure_fresh_start()

    def ensure_fresh_start(self):
        """
        Removes existing json file if it exists to start fresh.
        """
        if os.path.isfile(self.file_path):
            os.remove(self.file_path)
            
    def update_report(self, scores_dict, ball_poss_dict):
        report_dict = {key: {} for key in scores_dict.keys()}
        
        for key, value in scores_dict.items():
            report_dict[key]['score'] = value
        
        for key, value in ball_poss_dict.items():
            report_dict[key]['time'] = value['time']
            
        # Writing my_dict to a file in JSON format
        with open(self.file_path, 'w') as file:
            json.dump(report_dict, file, indent=4)
        

# Setup stage =================================================================
# HSV ranges per class --------------------------------------------------------
class HSVRangeSetup:
    def __init__(self, config):
        self.config = config
        self.n_classes = config['n_classes']
        self.labels_of_interest = config['player_labels']

    def extract_and_detect_frames(self, video_path, object_detector, labels_of_interest):
        """
        Randomly selects and processes 5 frames from the video, displaying and collecting crops
        from detections for specified labels of interest.
        """
        detected_crops = []
        cap = None  # Initialize cap variable to None
        try:
            if isinstance(video_path, int) or (isinstance(video_path, str) and video_path.isdigit()):
                print(video_path)
                cap = cv2.VideoCapture(int(video_path), cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(video_path, cv2.CAP_DSHOW)
            if not cap.isOpened():
                raise ValueError("Error: Could not open video.")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames > 0:
                random_frames = sorted(random.sample(range(total_frames), min(5, total_frames)))
            else:
                # If total_frames is 0 or negative (e.g., for camera feed), process the first 5 frames
                random_frames = range(5)

            for frame_index in tqdm(random_frames, desc="Detecting players"):
                if total_frames > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = cap.read()
                if not ret:
                    continue  # Skip if the frame could not be read

                # Detect objects in the frame
                detected_objects = object_detector.detect(frame)
                for det in detected_objects:
                    if det.label in labels_of_interest:
                        xmin, ymin, xmax, ymax = [int(x) for x in det.bbox]
                        crop = frame[ymin:ymax, xmin:xmax]
                        detected_crops.append(crop)
        except Exception as e:
            print(f"Error: {str(e)}")
        finally:
            if cap is not None:  # Check if cap is not None before releasing
                cap.release()

        return detected_crops
    # Function to resize and concatenate crops in a grid format
    def scale_and_concat_crops(self, crops, max_crops_per_row=10, max_crops_per_col=5, display_height=300, display_width=800):
        if not crops:
            return np.zeros((display_height, display_width, 3), dtype=np.uint8)
        
        # Calculate the number of rows needed for the grid
        num_rows = min(math.ceil(len(crops) / max_crops_per_row), max_crops_per_col)
    
        # Initialize a blank image to hold all crops
        grid_image = np.zeros((display_height, display_width, 3), dtype=np.uint8)
    
        # Calculate the size for each crop
        crop_height = display_height // num_rows
        crop_width = display_width // max_crops_per_row
    
        # Store the starting y-coordinate for the current row
        current_y = 0
    
        for i, crop in enumerate(crops):
            # Resize crop to fit within the grid cell
            resized_crop = cv2.resize(crop, (crop_width, crop_height))
    
            # Calculate the row and column for this crop
            row = i // max_crops_per_row
            col = i % max_crops_per_row
    
            # Calculate the starting x-coordinate for this crop
            current_x = col * crop_width
            current_y = row * crop_height
    
            # Place the resized crop into the grid image
            end_x = current_x + resized_crop.shape[1]
            end_y = current_y + resized_crop.shape[0]
            
            # Check if the crop can be placed without exceeding the grid's bounds
            if end_y <= grid_image.shape[0] and end_x <= grid_image.shape[1]:
                grid_image[current_y:end_y, current_x:end_x] = resized_crop
            else:
                # If the bounds are exceeded, break the loop
                break
    
        return grid_image
    
    def generate_unique_id(self, lower_bound, upper_bound):
        """Generates a unique ID from HSV lower and upper bounds."""
        return '_'.join(map(str, np.concatenate((lower_bound, upper_bound))))
    
    def save_hsv_ranges(self, hsv_ranges_dict, file_path):
        """Saves the HSV ranges dictionary to a file in JSON format."""
        with open(file_path, 'w') as file:
            json.dump(hsv_ranges_dict, file, indent=4)
    
    def load_hsv_ranges(self, file_path):
        """Loads the HSV ranges from a file, converts them back to numpy arrays."""
        with open(file_path, 'r') as file:
            hsv_ranges_dict = json.load(file)
        for unique_id in hsv_ranges_dict:
            hsv_ranges_dict[unique_id]['lower_bound'] = np.array(hsv_ranges_dict[unique_id]['lower_bound'])
            hsv_ranges_dict[unique_id]['upper_bound'] = np.array(hsv_ranges_dict[unique_id]['upper_bound'])
            hsv_ranges_dict[unique_id]['bgr_color'] = tuple(map(int, hsv_ranges_dict[unique_id]['bgr_color']))
        return hsv_ranges_dict
    
    def setup_hsv_ranges(self, object_detector):
        if os.path.exists(self.config.get('hsv_ranges_path', '')):
            print("HSV ranges file found. Loading HSV ranges...")
            return self.load_hsv_ranges(self.config['hsv_ranges_path'])

        detected_crops = self.extract_and_detect_frames(self.config['input_video_path'],
                                                        object_detector,
                                                        self.labels_of_interest)

        concatenated_crops = self.scale_and_concat_crops(detected_crops)
        segmentation = VideoSegmentation('Segmentation')
        hsv_ranges_dict = {}
        alphabet = iter(string.ascii_lowercase)  # Create an iterator over the alphabet

        print("Instructions:")
        print("- Press 'y' if you are satisfied with the HSV range and want to move to the next class.")
        print("- Press 'Esc' to exit the program.")

        for _ in range(self.n_classes):
            print(f"Setting up HSV range for class {_ + 1}/{self.n_classes}")
            while True:
                lower_bound, upper_bound = segmentation.update_segmentation(concatenated_crops)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('y'):
                    unique_id = self.generate_unique_id(lower_bound, upper_bound)
                    # Calculate the midpoint of the hue range for RGB color
                    mid_hue = (lower_bound[0] + upper_bound[0]) / 2
                    mid_hsv_color = np.array([[[int(mid_hue), 255, 255]]], dtype=np.uint8)
                    mid_rgb_color = cv2.cvtColor(mid_hsv_color, cv2.COLOR_HSV2BGR)[0][0].tolist()
                    
                    # Assign a letter from the alphabet to this unique_id
                    team_letter = next(alphabet, None)
                    if team_letter is None:
                        raise ValueError("Exceeded alphabet limit for unique team letters.")
                    
                    hsv_ranges_dict[unique_id] = {
                        'lower_bound': lower_bound.tolist(),
                        'upper_bound': upper_bound.tolist(),
                        'bgr_color': mid_rgb_color,
                        'team_letter': team_letter
                    }
                    break
                elif key == 27:  # Escape key
                    print("Exiting the program.")
                    cv2.destroyAllWindows()
                    sys.exit()

            segmentation.reset_trackbars()

        cv2.destroyAllWindows()
        self.save_hsv_ranges(hsv_ranges_dict, self.config['hsv_ranges_path'])
        return self.load_hsv_ranges(self.config['hsv_ranges_path'])

class VideoSegmentation:
    def __init__(self, window_name):
        self.window_name = window_name
        cv2.namedWindow(self.window_name)
        self.lower_bound = np.array([0, 0, 0])
        self.upper_bound = np.array([179, 255, 255])
        self.create_trackbars()

    def create_trackbars(self):
        cv2.createTrackbar('H Min', self.window_name, 0, 179, self.noop)
        cv2.createTrackbar('H Max', self.window_name, 179, 179, self.noop)
        cv2.createTrackbar('S Min', self.window_name, 0, 255, self.noop)
        cv2.createTrackbar('S Max', self.window_name, 255, 255, self.noop)
        cv2.createTrackbar('V Min', self.window_name, 0, 255, self.noop)
        cv2.createTrackbar('V Max', self.window_name, 255, 255, self.noop)

    def noop(self, x):
        pass

    def get_hsv_ranges(self):
        return self.lower_bound, self.upper_bound

    def update_segmentation(self, concatenated_frame):
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
        cv2.setTrackbarPos('H Min', self.window_name, 0)
        cv2.setTrackbarPos('H Max', self.window_name, 179)
        cv2.setTrackbarPos('S Min', self.window_name, 0)
        cv2.setTrackbarPos('S Max', self.window_name, 255)
        cv2.setTrackbarPos('V Min', self.window_name, 0)
        cv2.setTrackbarPos('V Max', self.window_name, 255)
        
        
# Manual homography matrix ----------------------------------------------------
class HomographySetup:
    def __init__(self, config):
        self.config = config
        self.layout_img, self.first_frame = self.load_and_prepare_images()
        self.points_layout = []
        self.points_frame = []

    def load_and_prepare_images(self):
        layout_img = cv2.imread(self.config['input_layout_image'])
        cap = cv2.VideoCapture(self.config['input_video_path'], cv2.CAP_DSHOW)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print("Error: Could not read the video frame.")
            return None, None

        return layout_img, frame

    def click_event(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            concatenated_img, max_width = params
            if x < max_width:  # Clicked on the layout image
                self.points_layout.append((x, y))
            else:  # Clicked on the video frame
                self.points_frame.append((x - max_width, y))
            self.update_display(concatenated_img, max_width)

    def update_display(self, concatenated_img, max_width):
        concatenated_img[:, :] = np.concatenate((self.padded_layout_img, self.padded_first_frame), axis=1)
        for pt_layout in self.points_layout:
            cv2.circle(concatenated_img, pt_layout, 5, (255, 0, 0), -1)
        for pt_frame in self.points_frame:
            cv2.circle(concatenated_img, (pt_frame[0] + max_width, pt_frame[1]), 5, (0, 255, 0), -1)
        for pt_layout, pt_frame in zip(self.points_layout, self.points_frame):
            cv2.line(concatenated_img, pt_layout, (pt_frame[0] + max_width, pt_frame[1]), (0, 0, 255), 2)
        cv2.imshow("Homography Points Selection", concatenated_img)

    def compute_homography_matrix(self):
        if self.config['h_matrix_path'] and os.path.exists(self.config['h_matrix_path']):
            return np.load(self.config['h_matrix_path'])

        self.padded_layout_img, self.padded_first_frame, concatenated_img = self.prepare_images_for_display()
        max_width = max(self.layout_img.shape[1], self.first_frame.shape[1])

        cv2.namedWindow("Homography Points Selection", cv2.WINDOW_NORMAL)
        cv2.imshow("Homography Points Selection", concatenated_img)
        cv2.setMouseCallback("Homography Points Selection", self.click_event, (concatenated_img, max_width))

        print("Instructions:")
        print("- Click corresponding points on the layout image and the video frame.")
        print("- Press 'y' to confirm and calculate homography.")
        print("- Press 'Esc' to quit.")
        print("- Press 'r' to remove the last point match.")

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # Esc key
                cv2.destroyAllWindows()
                return None  # Return early since homography cannot be computed
            elif key == ord('y'):
                if len(self.points_layout) >= 4 and len(self.points_frame) >= 4:
                    H, _ = cv2.findHomography(np.array(self.points_frame), np.array(self.points_layout))
                    if self.config['h_matrix_path']:
                        np.save(self.config['h_matrix_path'], H)
                    cv2.destroyAllWindows()
                    return H
                else:
                    print("Not enough points to compute homography.")
                    cv2.destroyAllWindows()
                    return None  # Return early since homography cannot be computed
            elif key == ord('r') and self.points_layout and self.points_frame:  # Remove the last point match
                self.points_layout.pop()
                self.points_frame.pop()
                self.update_display(concatenated_img, max_width)

    def prepare_images_for_display(self):
        max_height = max(self.layout_img.shape[0], self.first_frame.shape[0])
        max_width = max(self.layout_img.shape[1], self.first_frame.shape[1])

        padded_layout_img = cv2.copyMakeBorder(self.layout_img, 0, max_height - self.layout_img.shape[0], 0, max_width - self.layout_img.shape[1], cv2.BORDER_CONSTANT, value=[0, 0, 0])
        padded_first_frame = cv2.copyMakeBorder(self.first_frame, 0, max_height - self.first_frame.shape[0], 0, max_width - self.first_frame.shape[1], cv2.BORDER_CONSTANT, value=[0, 0, 0])

        concatenated_img = np.concatenate((padded_layout_img, padded_first_frame), axis=1)
        return padded_layout_img, padded_first_frame, concatenated_img
    
# Object detection functions ==================================================
class DetectedObject:
    def __init__(self, bbox, lbl):
        self.bbox = [int(j) for j in bbox]
        self.label = lbl
    def get_bbox_bottom(self):
        xmin, ymin, xmax, ymax = self.bbox
        return np.array([int((xmin + xmax) / 2), int(ymax)])
        
class ObjectDetector:
    def __init__(self, config):
        self.model = YOLO(config["yolo_model_path"])
        
    def detect(self, frame, verbose=False, conf=0.7, imgsz=640):
        yolo_detections = self.model.predict(frame, verbose=verbose, conf=conf, imgsz=imgsz, task="detect")
        detected_objects = self.compute_detected_objects(yolo_detections)
        
        return detected_objects

    def compute_detected_objects(self, yolo_detections):
        detected_objects = []
        detections_as_xyxy = yolo_detections[0].boxes.xyxy
        labels = yolo_detections[0].boxes.cls

        for lbl, det_xyxy in zip(labels, detections_as_xyxy):
            det_xyxy = det_xyxy.cpu().numpy()
            lbl = int(lbl.cpu().numpy())

            x1, y1, x2, y2 = det_xyxy
            det_object = DetectedObject(det_xyxy, lbl)
            detected_objects.append(det_object)
        
        return detected_objects
    
# Classes os elements of interest =============================================
# Manual goal polygon ---------------------------------------------------------
class GoalPolygon:
    def __init__(self, config, teams_dict):
        self.config = config
        self.first_frame = self.load_first_frame()
        self.polygon = []
        self.load_and_draw_polygon()
        self.ball_in = False
        
        self.teams_dict, self.scores_dict = self.initialize_teams_scores(teams_dict)
        
    def initialize_teams_scores(self, teams_dict):
        scores_dict = {}
        
        for key in teams_dict.keys():
            teams_dict[key]['score'] = 0
            
            team_letter = teams_dict[key]['team_letter']
            scores_dict[team_letter] = 0
            
        return teams_dict, scores_dict
        
    def load_polygon_if_exists(self):
        polygon_path = self.config['goal_polygon_path']
        if os.path.exists(polygon_path):
            self.polygon = np.load(polygon_path)
            return True
        return False

    def draw_polygon(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.polygon.append((x, y))
            cv2.circle(self.first_frame, (x, y), 5, (0, 255, 0), -1)
        if len(self.polygon) > 1:
            cv2.polylines(self.first_frame, [np.array(self.polygon)], False, (255, 0, 0), 2)
        cv2.imshow("Goal Polygon", self.first_frame)
    
    def load_first_frame(self):
        cap = cv2.VideoCapture(self.config['input_video_path'], cv2.CAP_DSHOW)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print("Error loading the first frame.")
            return None
        return frame
    
    def load_and_draw_polygon(self):
        if not self.load_polygon_if_exists():
            cv2.namedWindow("Goal Polygon")
            cv2.setMouseCallback("Goal Polygon", self.draw_polygon)
    
            print("Instructions:")
            print("- Draw the goal polygon. Click to add points.")
            print("- Press 'y' to save and continue.")
            print("- Press 'Esc' to exit without saving.")
            print("- Press 'r' to remove the last point.")
    
            while True:
                cv2.imshow("Goal Polygon", self.first_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('y'):
                    # Close the polygon if it's not already closed
                    if len(self.polygon) > 2 and self.polygon[0] != self.polygon[-1]:
                        self.polygon.append(self.polygon[0])
                    self.save_polygon()
                    break
                elif key == 27:  # Esc key
                    print("Exiting without saving the polygon.")
                    cv2.destroyAllWindows()
                elif key == ord('r') and self.polygon:  # Remove the last point
                    self.polygon.pop()
                    self.first_frame = self.load_first_frame()  # Reload the frame to clear previous drawings
                    if len(self.polygon) > 1:
                        cv2.polylines(self.first_frame, [np.array(self.polygon)], False, (0, 255, 0), 2)
                    for point in self.polygon:
                        cv2.circle(self.first_frame, point, 5, (0, 255, 0), -1)
    
            cv2.destroyAllWindows()

    def save_polygon(self):
        polygon_path = self.config['goal_polygon_path']
        np.save(polygon_path, np.array(self.polygon))
        
    def draw_polygon_on_frame(self, frame):
        # Check if polygon has points
        if len(self.polygon) > 0:
            # Convert polygon points to a numpy array with shape (n, 1, 2) for cv2.polylines compatibility
            pts = np.array(self.polygon, np.int32)
            pts = pts.reshape((-1, 1, 2))
            # Draw the polygon on the frame
            
            if self.ball_in:
                cv2.fillPoly(frame, [pts], color=(0, 255, 0))
            else:
                cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=3)
            
    def update_draw_score(self, ball_object, frame):
        ball_center = ball_object.center_point
        
        if ball_center is not None:
            pts = np.array(self.polygon, np.int32)
            result = int(cv2.pointPolygonTest(pts, ball_center, False))
            
            if not self.ball_in and result == 1:
                self.ball_in = True
                
                team_id = ball_object.last_team_id
                
                if team_id in self.teams_dict:
                    self.teams_dict[team_id]['score'] += 1
                    self.scores_dict[self.teams_dict[team_id]['team_letter']] = self.teams_dict[team_id]['score']
            if self.ball_in and result == -1:
                self.ball_in = False
            
        self.draw_score_box(frame)
    
    def draw_score_box(self, frame):
        # Starting position for the score boxes
        offset_x = 10  # Offset from the left edge
        offset_y = 10  # Offset from the top edge
    
        # "SCORE" label dimensions
        score_label_height = 20  # Reduced height to make it thinner
    
        # Score box dimensions
        box_width = 100
        box_height = 50
        label_box_height = 20
    
        # Calculate the total width of all score boxes
        num_teams = len(self.teams_dict)
        total_score_width = num_teams * box_width
    
        # Draw the black rectangle for "SCORE" label across all score boxes
        cv2.rectangle(frame, (offset_x, offset_y), (offset_x + total_score_width, offset_y + score_label_height), (0, 0, 0), -1)
    
        # Add "SCORE" text on the rectangle, centered
        score_label_text = "SCORE"
        score_label_font_scale = 0.5
        score_label_font_thickness = 2
        score_label_text_size = cv2.getTextSize(score_label_text, cv2.FONT_HERSHEY_SIMPLEX, score_label_font_scale, score_label_font_thickness)[0]
        score_label_text_x = offset_x + (total_score_width - score_label_text_size[0]) // 2
        score_label_text_y = offset_y + score_label_height - (score_label_height - score_label_text_size[1]) // 2
        cv2.putText(frame, score_label_text, (score_label_text_x, score_label_text_y), cv2.FONT_HERSHEY_SIMPLEX, score_label_font_scale, (255, 255, 255), score_label_font_thickness)
    
        # Adjust initial top left y position for the team score boxes to be directly below the "SCORE" label
        initial_top_left_y = offset_y + score_label_height  # No additional space needed
    
        for index, (team_id, team_info) in enumerate(self.teams_dict.items()):
            # Calculate positions for each team's score box
            top_left_x = offset_x + (box_width * index)
            top_left_y = initial_top_left_y
            bottom_right_x = top_left_x + box_width
            label_top_left_y = top_left_y
            label_bottom_right_y = top_left_y + label_box_height
    
            team_letter = team_info['team_letter']
            bgr_color = team_info['bgr_color']
            score = team_info['score']
    
            # Draw the label box with team-specific BGR color
            cv2.rectangle(frame, (top_left_x, label_top_left_y), (bottom_right_x, label_bottom_right_y), bgr_color, -1)
    
            # Draw the white score box
            cv2.rectangle(frame, (top_left_x, label_bottom_right_y), (bottom_right_x, label_bottom_right_y + box_height), (255, 255, 255), -1)
    
            # Team letter in the label box, centered
            label_text = f"({team_letter.upper()})"
            label_font_scale = 0.5
            label_text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, 1)[0]
            label_text_x = top_left_x + (box_width - label_text_size[0]) // 2
            label_text_y = label_top_left_y + (label_box_height + label_text_size[1]) // 2 - 5  # Adjust vertical position
            cv2.putText(frame, label_text, (label_text_x, label_text_y), cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, (255, 255, 255), 2)
    
            # Score text in the score box, centered
            score_text = f"{score}"
            score_font_scale = 1
            score_text_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, score_font_scale, 2)[0]
            score_text_x = top_left_x + (box_width - score_text_size[0]) // 2
            score_text_y = label_bottom_right_y + (box_height + score_text_size[1]) // 2 - 5  # Adjust vertical position
            cv2.putText(frame, score_text, (score_text_x, score_text_y), cv2.FONT_HERSHEY_SIMPLEX, score_font_scale, (0, 0, 0), 2)
    
# Team Player class -----------------------------------------------------------
class TeamPlayer:
    def __init__(self, config, team_id, team_dict):
        self.hsv_classes_ids, self.hsv_classes_ranges = self.extract_hsv_classes_ranges(team_dict)
        
        self.team_id = team_id
        self.team_letter = team_dict[self.team_id]['team_letter']
        self.color = team_dict[self.team_id]['bgr_color']
        self.max_bbox = None
        self.center_point = None
        self.labels_of_interest = config['player_labels']
        
    def extract_hsv_classes_ranges(self, team_ranges):
        keys_to_keep = ['lower_bound', 'upper_bound']
        
        hsv_ids_list = []
        hsv_ranges_list = []
    
        for class_id, nested_dict in team_ranges.items():
            # Directly append the class_id to the ids list
            hsv_ids_list.append(class_id)
            
            # Extract the desired keys if they exist, and append as a pair to the ranges list
            if all(key in nested_dict for key in keys_to_keep):
                hsv_ranges_list.append([nested_dict[keys_to_keep[0]], nested_dict[keys_to_keep[1]]])
        
        return hsv_ids_list, hsv_ranges_list
    
    def classify_by_color(self, frame_crop):
        hsv_classes_ids, hsv_classes_ranges = self.hsv_classes_ids, self.hsv_classes_ranges
        
        frame_crop_hsv = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2HSV)
        class_scores = np.zeros((len(hsv_classes_ids),), dtype=np.float32)

        for i, hsv_ranges in enumerate(hsv_classes_ranges):
            lower_bound, upper_bound = hsv_ranges
            mask = cv2.inRange(frame_crop_hsv, lower_bound, upper_bound)

            class_scores[i] = np.sum(mask)

        return hsv_classes_ids[np.argmax(class_scores)]
    
    def filter_detections_by_class(self, detected_bbox, frame):
        filtered_bbox = []
        
        target_id = self.team_id
        
        for bbox in detected_bbox:
            x1, y1, x2, y2 = bbox
            frame_crop = frame[int(y1):int(y2)+1, int(x1):int(x2)+1, :]
            pred_id = self.classify_by_color(frame_crop)
            
            if pred_id == target_id:
                filtered_bbox.append(bbox)
            
        return filtered_bbox
    
    def update_draw_location(self, detected_objects, frame):
        detected_bbox = [obj.bbox for obj in detected_objects if obj.label in self.labels_of_interest]
        
        if len(detected_bbox) > 0:
            # Filter only bboxes of current player team class
            filtered_bbox = self.filter_detections_by_class(detected_bbox, frame)
            
            if len(filtered_bbox) > 0:
                bbox_areas = np.array([(box[2] - box[0]) * (box[3] - box[1]) for box in filtered_bbox])
                
                index_max_area = np.argmax(bbox_areas)
                self.max_bbox = filtered_bbox[index_max_area]
                self.center_point = (int((self.max_bbox[0] + self.max_bbox[2]) / 2), int(self.max_bbox[3]))
                self.draw_object(self.max_bbox, frame)
            else:
                self.max_bbox = None
                self.center_point = None
        else:
            self.max_bbox = None
            self.center_point = None

    def draw_object(self, bbox, frame):
        color = self.color
        text = f'Team {self.team_letter.upper()}'
        
        xmin, ymin, xmax, ymax = [int(x) for x in bbox]
        # Define colors and font for the bounding box and label
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 2

        # Draw the bounding box
        frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        
        # Draw bottom center point
        frame = cv2.circle(frame, self.center_point, radius=5, color=color, thickness=-1)

        # Calculate text size for background
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        
        # Draw background for text for better visibility
        frame = cv2.rectangle(frame, (xmin, ymin - 20), (xmin + text_width, ymin), color, -1)

        # Put the text (label) on the frame
        frame = cv2.putText(frame, text, (xmin, ymin - 5), font, font_scale, (255, 255, 255), font_thickness)
        
# Ball class ------------------------------------------------------------------
class Ball:
    def __init__(self, config):
        self.color = (255, 165, 0)
        self.last_team_id = None
        self.last_team_letter = None
        self.center_point = None
        
        self.labels_of_interest = config['ball_labels']
        
    def update_draw_location(self, players_obj, detected_objects, frame):
        detected_bbox = [obj.bbox for obj in detected_objects if obj.label in self.labels_of_interest]
        
        if len(detected_bbox) > 0:
            bbox_areas = np.array([(box[2] - box[0]) * (box[3] - box[1]) for box in detected_bbox])
            
            index_max_area = np.argmax(bbox_areas)
            max_bbox = detected_bbox[index_max_area]
            center_x = int((max_bbox[0] + max_bbox[2]) / 2)
            center_y = int((max_bbox[1] + max_bbox[3]) / 2)
            
            self.center_point = (center_x, center_y)
            
            for player in players_obj:
                if player.max_bbox is not None:
                    px1, py1, px2, py2 = player.max_bbox
                    
                    if (center_x > px1) and (center_x < px2):
                        if (center_y > py1 + 2*py2/3) and (center_y < py2):
                            self.color = player.color
                            self.last_team_id = player.team_id
                            self.last_team_letter = player.team_letter
                    
            self.draw_object(max_bbox, frame)
        else:
            self.center_point = None
            
    def draw_object(self, bbox, frame):
        color = self.color
        
        # Calculate the center and size of the bounding box
        center_x = int((bbox[0] + bbox[2]) / 2)
        center_y = int((bbox[1] + bbox[3]) / 2)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
    
        # Determine the radius as half the smaller of the width or height
        radius = int(min(width, height) / 2)
    
        # Draw the circle
        cv2.circle(frame, (center_x, center_y), radius, color, -1)
        
# Layout projector object ------------------------------------------------------
class LayoutProjector:
    def __init__(self, config, H, teams_dict):
        self.layout_image, self.layout_img_gray = self.load_layout(config['input_layout_image'])
        self.heatmaps_dict, self.overlay_heatmaps_dict = self.initialize_heatmaps_dict(teams_dict)
        
        self.H = H
        self.layout_dict, self.ball_poss_dict = self.initialize_layout_dict(teams_dict)
        self.fps = 0
    
    def reset_layout_dict(self):
        # Reset layout_dict to its initial state
        self.layout_dict = {"a": [], "b": [], "ball": [], "ball_possession": []}
    def initialize_heatmaps_dict(self, teams_dict):
        heatmap_zeros = np.zeros((self.layout_image.shape[0], self.layout_image.shape[1]), dtype=np.float32)
        heatmaps_dict = {teams_dict[key]['team_letter']: heatmap_zeros.copy() for key in teams_dict.keys()}
        
        heatmap_zeros_3c = np.zeros((self.layout_image.shape[0], self.layout_image.shape[1], 3), dtype=np.float32)
        overlay_heatmaps_dict = {teams_dict[key]['team_letter']: heatmap_zeros_3c.copy() for key in teams_dict.keys()}
        
        return heatmaps_dict, overlay_heatmaps_dict
        
    def load_layout(self, img_path):
        layout_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        
        layout_img_gray = cv2.cvtColor(layout_img, cv2.COLOR_BGR2GRAY)
        layout_img_gray = cv2.cvtColor(layout_img_gray, cv2.COLOR_GRAY2BGR)
        
        return layout_img, layout_img_gray    
    
    def initialize_layout_dict(self, teams_dict):
        ball_poss_dict = {teams_dict[key]['team_letter']: {'color': teams_dict[key]['bgr_color'], 'frames_count': 0, 'time': '00:00'} for key in teams_dict.keys()}
        
        layout_dict = {teams_dict[key]['team_letter']: [] for key in teams_dict.keys()}
        layout_dict['ball'] = []
        layout_dict['ball_possession'] = []
        
        return layout_dict, ball_poss_dict
    
    def convert_frames_to_time(self):
        for value in self.ball_poss_dict.values():
            frame_count = value['frames_count']
            current_time = frame_count / self.fps  # Time in seconds
            minutes = int(current_time // 60)
            seconds = int(current_time % 60)
            time_str = f"{minutes:02d}:{seconds:02d}"  # Format time as mm:ss
            
            value['time'] = time_str
    
    def update_draw_layout_dict(self, team_players_list, ball_object):
        temp_layout_dict = {}
        
        for player in team_players_list:
            temp_layout_dict[player.team_letter] = {'point': self.apply_homography_to_point(player.center_point),
                                                    'color': player.color}
            
        temp_layout_dict['ball'] = {'point':self.apply_homography_to_point(ball_object.center_point),
                                    'color': ball_object.color}
        
        for key in temp_layout_dict.keys():
            self.layout_dict[key].append(temp_layout_dict[key]['point'])
        
        if ball_object.last_team_letter is not None:
            self.ball_poss_dict[ball_object.last_team_letter]['frames_count'] += 1
            self.convert_frames_to_time()
            
        self.layout_dict['ball_possession'].append(ball_object.last_team_letter)
        
        drawn_layout = self.draw_transformed_points_with_heatmap(temp_layout_dict)
        
        return drawn_layout, self.overlay_heatmaps_dict
        
    def draw_transformed_points_with_heatmap(self, temp_layout_dict):
        layout_img = self.layout_image.copy()
        
        # Draw a circle with a black border
        border_thickness = 3
        circle_radius = 10
        
        for key in temp_layout_dict.keys():
            point = temp_layout_dict[key]['point']
            
            if point is not None:
                x, y = int(point[0]), int(point[1])
                
                if key == 'ball':
                    border_color = (255, 255, 255)
                else:
                    border_color = (0, 0, 0)
                    
                    # Update heatmaps
                    mask = np.zeros((layout_img.shape[0], layout_img.shape[1]), dtype=np.float32)
                    cv2.circle(mask, (x, y), circle_radius, (1,), thickness=-1)
                    
                    self.heatmaps_dict[key] += mask
                    self.visualize_heatmaps()
                
                # Draw points over layout
                circle_color = temp_layout_dict[key]['color']
                
                cv2.circle(layout_img, (x, y), circle_radius + border_thickness, border_color, thickness=-1)
                cv2.circle(layout_img, (x, y), circle_radius, circle_color, thickness=-1)
            
        return layout_img
    
    def visualize_heatmaps(self, base=10, alpha=0.5):
        for key, heatmap in self.heatmaps_dict.items():
            heatmap_log = np.log1p(heatmap) / np.log(base)
            
            # Normalize the heatmap for display
            heatmap_normalized = cv2.normalize(heatmap_log, None, 0, 255, cv2.NORM_MINMAX)
            heatmap_uint8 = np.uint8(heatmap_normalized)
            
            # Apply a colormap for visualization
            heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            
            overlayed_image = cv2.addWeighted(self.layout_img_gray, 1 - alpha, heatmap_colored, alpha, 0)
            
            self.overlay_heatmaps_dict[key] = overlayed_image
            
        
    def apply_homography_to_point(self, point):
        if point is not None:
            point = np.array(point)
            H = self.H 
            
            # Convert point to homogeneous coordinates
            point_homogeneous = np.append(point, 1)
            
            # Apply the homography matrix
            point_transformed_homogeneous = np.dot(H, point_homogeneous)
            
            # Convert back to Cartesian coordinates
            point_transformed = point_transformed_homogeneous[:2] / point_transformed_homogeneous[2]
               
            return point_transformed
        else:
            return point
        
    def update_draw_possession_time(self, frame):
        # Calculate the dimensions of the frame and the text boxes
        frame_height, frame_width = frame.shape[:2]
        offset_x = 10  # Offset from the right edge
        offset_y = 10  # Offset from the top edge
    
        # "POSSESSION" label dimensions to match "SCORE" label
        label_height = 20
        box_width = 100
        box_height = 50
        label_box_height = 20
        font_scale = 0.5
        font_thickness = 2
    
        # Calculate the total width of the possession boxes
        num_teams = len(self.ball_poss_dict)
        total_possession_width = num_teams * box_width
    
        # Draw the "POSSESSION" label on the top right
        possession_label_top_right_x = frame_width - offset_x - total_possession_width
        cv2.rectangle(frame, (possession_label_top_right_x, offset_y), (frame_width - offset_x, offset_y + label_height), (0, 0, 0), -1)
        label_text = "POSSESSION"
        label_text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        label_text_x = possession_label_top_right_x + (total_possession_width - label_text_size[0]) // 2
        label_text_y = offset_y + label_height - (label_height - label_text_size[1]) // 2
        cv2.putText(frame, label_text, (label_text_x, label_text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
    
        # Initialize the y-coordinate for the team boxes to be below the "POSSESSION" label
        initial_top_left_y = offset_y + label_height
    
        for index, (team_letter, values) in enumerate(sorted(self.ball_poss_dict.items(), key=lambda x: x[0], reverse=True)):
            color = values['color']
            time_str = values['time']
            
            # Draw the team-specific possession time boxes
            top_right_x = frame_width - offset_x - (box_width * (index + 1))
            cv2.rectangle(frame, (top_right_x, initial_top_left_y), (top_right_x + box_width, initial_top_left_y + label_box_height), color, -1)
    
            # Team letter in the label box, centered
            team_label = f"({team_letter.upper()})"
            team_label_size = cv2.getTextSize(team_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            team_label_x = top_right_x + (box_width - team_label_size[0]) // 2
            team_label_y = initial_top_left_y + (label_box_height + team_label_size[1]) // 2 - 5
            cv2.putText(frame, team_label, (team_label_x, team_label_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
    
            # Draw the white box for the possession time, below the team letter box
            cv2.rectangle(frame, (top_right_x, initial_top_left_y + label_box_height), (top_right_x + box_width, initial_top_left_y + label_box_height + box_height), (255, 255, 255), -1)
    
            # Possession time in the white box, centered
            time_text_size = cv2.getTextSize(time_str, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            time_text_x = top_right_x + (box_width - time_text_size[0]) // 2
            time_text_y = initial_top_left_y + label_box_height + (box_height + time_text_size[1]) // 2 - 5
            cv2.putText(frame, time_str, (time_text_x, time_text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
        return frame

# Video processor class =======================================================
class VideoProcessor:
    def __init__(self, config, object_detector, goal_polygon, team_players_list, ball_object, layout_projector, database_writer, report_writer):
        self.database_writer = database_writer
        self.report_writer = report_writer
        
        self.config = config
        self.object_detector = object_detector
        
        self.goal_polygon = goal_polygon
        self.team_players_list = team_players_list
        self.ball_object = ball_object
        self.layout_projector = layout_projector
        
        self.fps = 0
        self.frame_count = 0
    
    def process_frame(self, frame):
        self.frame_count += 1
        self.layout_projector.reset_layout_dict()
        detected_objects = self.object_detector.detect(frame)
        
        self.goal_polygon.draw_polygon_on_frame(frame)
        
        for player in self.team_players_list:
            player.update_draw_location(detected_objects, frame)
        
        self.ball_object.update_draw_location(self.team_players_list, detected_objects, frame)
        self.goal_polygon.update_draw_score(self.ball_object, frame)
        
        drawn_layout, overlay_heatmaps_dict = self.layout_projector.update_draw_layout_dict(self.team_players_list, self.ball_object)
        frame = self.layout_projector.update_draw_possession_time(frame)
        
        self.database_writer.update_db(self.layout_projector.layout_dict, self.frame_count, self.goal_polygon.scores_dict)
        
        
        self.report_writer.update_report(self.goal_polygon.scores_dict, self.layout_projector.ball_poss_dict)

        return frame, drawn_layout, overlay_heatmaps_dict
        
    def process_video(self):
        cap = cv2.VideoCapture(self.config['input_video_path'], cv2.CAP_DSHOW)
    
        # Check if video opened successfully
        if not cap.isOpened():
            print("Error opening video stream or file")
            return
        
        # Get video frame width, height, and FPS for the output video
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.layout_projector.fps = self.fps
        
        # Define the codec and create VideoWriter object
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' for .mp4 files
        out = cv2.VideoWriter(self.config['output_video_path'], self.fourcc, self.fps, (frame_width, frame_height))
        heatmap_outs_dict, layout_out_writer = self.initialize_heatmap_layout_output_writers()
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        with tqdm(total=total_frames, desc="Processing video frames") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    processed_frame, drawn_layout, overlay_heatmaps_dict = self.process_frame(frame)
                    
                    out.write(processed_frame)
                    layout_out_writer.write(drawn_layout)
    
                    # Display the single composite window
                    cv2.imshow("Detections", processed_frame)
                    cv2.imshow("Layout", drawn_layout)
                    
                    for key, heatmap in overlay_heatmaps_dict.items():
                        if heatmap.dtype != np.uint8:
                            heatmap = cv2.convertScaleAbs(heatmap)
                        heatmap_outs_dict[key].write(heatmap)
                        cv2.imshow(key, heatmap)
                    
                    # Check if 'ESC' key was pressed
                    if cv2.waitKey(1) & 0xFF == 27:  # 27 is the ASCII value for 'ESC'
                        break
    
                    pbar.update(1)

                else:
                    break

    	# Release everything if job is finished
        cap.release()
        out.release()
        layout_out_writer.release()
        
        for out_wirter in heatmap_outs_dict.values():
            out_wirter.release()
        
        cv2.destroyAllWindows()
        
        # Save final heatmaps
        for key, heatmap in overlay_heatmaps_dict.items():
            cv2.imwrite(config['output_image_heatmaps'][key], heatmap)

    def send_processed_video_frames(self, host='0.0.0.0', port=8080):
        """
        Sends processed video frames over a network connection.
        Args:
            host (str, optional): The IP address or hostname to bind the server socket to. Defaults to '0.0.0.0'.
            port (int, optional): The port number to bind the server socket to. Defaults to 8080.
        """
        # Set up server socket for receiving client connections
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((host, port))
        server_socket.listen()
        print(f"Listening for client connections on {host}:{port}")
        # Accept client connection
        conn, addr = server_socket.accept()
        print(f"Connection from {addr}")
        # Open video capture
        input_video_path = self.config['input_video_path']
        if input_video_path == 0 or (isinstance(input_video_path, str) and input_video_path.isdigit()):
            cap = cv2.VideoCapture(int(input_video_path), cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(input_video_path, cv2.CAP_DSHOW)
        
        # Check if video opened successfully
        if not cap.isOpened():
            print("Error opening video stream or file")
            return
        
        # Get video frame width, height, and FPS for the output video
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.layout_projector.fps = self.fps
        
        # Define the codec and create VideoWriter object for output video
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.config['output_video_path'], self.fourcc, self.fps, (frame_width, frame_height))
        heatmap_outs_dict, layout_out_writer = self.initialize_heatmap_layout_output_writers()
        # Handling unknown total frame count for live camera feed
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else float('inf')
        
        # Process video frames
        with tqdm(total=None if total_frames == float('inf') else total_frames, desc="Processing video frames") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    processed_frame, drawn_layout, overlay_heatmaps_dict = self.process_frame(frame)
                    
                    # Write processed frame and layout to output video
                    out.write(processed_frame)
                    layout_out_writer.write(drawn_layout)    
                    
                    # Write overlay heatmaps to respective output videos
                    for key, heatmap in overlay_heatmaps_dict.items():
                        if heatmap.dtype != np.uint8:
                            heatmap = cv2.convertScaleAbs(heatmap)
                        heatmap_outs_dict[key].write(heatmap)
                    
                    # Resize frames and heatmaps for network transmission
                    fx, fy = 0.75, 0.75
                    processed_frame = cv2.resize(processed_frame, (0, 0), fx=fx, fy=fy)
                    drawn_layout = cv2.resize(drawn_layout, (0, 0), fx=fx, fy=fy)
                    overlay_heatmaps_dict = {key: cv2.resize(heatmap, (0, 0), fx=fx, fy=fy) for key, heatmap in overlay_heatmaps_dict.items()}
                    
                    # Encode frames as JPEG and send over network connection
                    JPEG_QUALITY = 90
                    _, jpeg_processed_frame = cv2.imencode('.jpg', processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                    _, jpeg_drawn_layout = cv2.imencode('.jpg', drawn_layout, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                    _, jpeg_heatmap_1 = cv2.imencode('.jpg', overlay_heatmaps_dict['a'], [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                    _, jpeg_heatmap_2 = cv2.imencode('.jpg', overlay_heatmaps_dict['b'], [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                    # Pack and send data over network connection
                    data = pickle.dumps((jpeg_processed_frame, jpeg_drawn_layout, jpeg_heatmap_1, jpeg_heatmap_2))
                    message = struct.pack("Q", len(data)) + data
                    conn.sendall(message)
                    pbar.update(1)
                else:
                    break
        # Release resources
        cap.release()
        out.release()
        layout_out_writer.release()
        conn.close()
        
        for out_writer in heatmap_outs_dict.values():
            out_writer.release()
        
        cv2.destroyAllWindows()
        
        # Save final heatmaps
        for key, heatmap in overlay_heatmaps_dict.items():
            cv2.imwrite(self.config['output_image_heatmaps'][key], heatmap)

    def initialize_heatmap_layout_output_writers(self):
        heatmap_outs_dict = {}
        
        heatmap_video_paths = self.config['output_video_heatmaps']
        layout_height, layout_width = self.layout_projector.layout_image.shape[0:2]
        
        for key, video_path in heatmap_video_paths.items():
            heatmap_outs_dict[key] = cv2.VideoWriter(video_path, self.fourcc, self.fps, (layout_width, layout_height))
            
        layout_out_writer = cv2.VideoWriter(self.config['layout_video_path'], self.fourcc, self.fps, (layout_width, layout_height))
        
        return heatmap_outs_dict, layout_out_writer
# Main function ===============================================================
if __name__ == "__main__":
    # Initialize the parser
    parser = argparse.ArgumentParser(description='Process video analysis with custom configuration.')

    # Adding arguments
    parser.add_argument('--input_video_path', type=str, default='0', help='Path to input video')
    parser.add_argument('--player_labels', nargs='+', type=int, default=[2, 3], help='Player labels')
    parser.add_argument('--ball_labels', nargs='+', type=int, default=[1], help='Ball labels')
    parser.add_argument('--n_classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--input_layout_image', type=str, default=r'C:\Users\shysk\Documents\soccer-analytics\Datasets\soccer field layout\soccer_field_layout.png', help='Path to input layout image')
    parser.add_argument('--yolo_model_path', type=str, default=r'C:\Users\shysk\Documents\soccer-analytics\Models\yolov8-demo-model\train\weights\nano\best.pt', help='Path to YOLO model')
    parser.add_argument('--output_base_dir', type=str, default=r'C:\Users\shysk\Documents\soccer-analytics\Usage\soccer-demo\outputs', help='Base directory for outputs')
    # Parse the arguments
    args = parser.parse_args()

    # 1. Setup ----------------------------------------------------------------
    # Update the config dictionary with arguments from the command line
    config = {
        'input_video_path': int(args.input_video_path) if args.input_video_path == '0' else args.input_video_path,
        'player_labels': args.player_labels,
        'ball_labels': args.ball_labels,
        'n_classes': args.n_classes,
        'input_layout_image': args.input_layout_image,
        'yolo_model_path': args.yolo_model_path,
        'output_base_dir': args.output_base_dir,
    }
    
    config = create_output_dirs(config)
    
    object_detector = ObjectDetector(config)
    
    # CSV writer
    database_writer = DatabaseWriter(config)
    
    # TXT Report Writer
    report_writer = ReportWriter(config)
    
    # HSV Ranges setup
    hsv_setup = HSVRangeSetup(config)
    teams_dict = hsv_setup.setup_hsv_ranges(object_detector)

    # Include the paths to the heatmap images and videos
    config = create_heatmaps_dirs(config, teams_dict)

    # H Matrix setup
    homography_setup = HomographySetup(config)
    H = homography_setup.compute_homography_matrix()
    
    # Create layout registration object
    layout_projector = LayoutProjector(config, H, teams_dict)
    
    #2. Analysis stage --------------------------------------------------------
    # Goal polygon setup
    goal_polygon = GoalPolygon(config, teams_dict)
    
    # Team players list
    team_players_list = [TeamPlayer(config, key, teams_dict) for key in teams_dict.keys()]
    
    # Create ball object
    ball_object = Ball(config)
    
    # Create video processing object and process video
    processor = VideoProcessor(config, object_detector, goal_polygon, team_players_list, ball_object, layout_projector, database_writer, report_writer)
    processor.send_processed_video_frames()
    # processor.process_video()
    # asyncio.run(processor.main())
