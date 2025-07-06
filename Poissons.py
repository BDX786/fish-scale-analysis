import cv2
import numpy as np
import os
from dataclasses import dataclass
from typing import List, Tuple
import csv
import datetime

@dataclass
class RingMeasurement:
    start_point: Tuple[int, int]
    end_point: Tuple[int, int]
    distance: float

class ScaleAnalyzer:
    def __init__(self, image_path, output_dir, csv_path):
        self.measurements: List[RingMeasurement] = []
        self.current_image = None
        self.processed_image = None
        self.scale_center = None
        self.measurement_line = None
        self.rings = []
        self.display_width = 800  # Width for display
        self.measurement_angle = 225  # Default angle for top-left direction (225 degrees)
        self.scale_contour = None
        self.ring_count = 0
        self.ring_positions = []
        self.scale_factor = 1.0  # Scale factor between display and original image
        self.angle_step = 2  # Smaller step for more precise angle adjustment
        self.measurement_points = []  # List to store the two points
        self.center_validated = False  # Flag for center validation
        self.point1_validated = False  # Flag for first point validation
        self.point2_validated = False  # Flag for second point validation
        self.current_state = "center"  # Current state: "center", "point1", "point2"
        self.current_image_path = None
        self.zoom_scale = 1.0
        self.zoom_center = None
        self.display_offset = (0, 0)  # For panning the zoomed image
        self.output_dir = "Output images"  # Added for save_results method
        self.csv_path = "scale_measurements.csv"  # Added for save_results method
        self.measurement_line_angle = 0  # Added for save_results method
        self.MAX_POINTS = 10  # Maximum number of points allowed
        self.point_validated = True  # Add this new flag
        self.pixels_per_mm = None  # Added for scale detection
        self.manual_mode = False  # Mode manuel pour la ligne de mesure
        self.manual_scale_mode = False  # Mode manuel pour l'échelle
        self.scale_points = []  # Points pour définir l'échelle manuellement
        self.SCALE_POINTS_NEEDED = 2  # Nombre de points nécessaires pour définir l'échelle
        self.scale_defined = False  # Pour vérifier si l'échelle est définie
        self.best_vertical = None  # Add this to track automatic detection
        self.best_lines = None  # Add this to track automatic detection
        
    def resize_image(self, image):
        # Calculate the height to maintain aspect ratio
        height, width = image.shape[:2]
        aspect_ratio = height / width
        new_height = int(self.display_width * aspect_ratio)
        
        # Resize base image
        base_resized = cv2.resize(image, (self.display_width, new_height))
        
        # Apply zoom if needed
        if self.zoom_scale != 1.0:
            try:
                # Calculate zoomed size
                zoom_width = int(self.display_width * self.zoom_scale)
                zoom_height = int(new_height * self.zoom_scale)
                
                # Resize to zoomed size
                zoomed = cv2.resize(base_resized, (zoom_width, zoom_height))
                
                # Calculate visible region
                x_offset, y_offset = self.display_offset
                visible_width = min(self.display_width, zoom_width)
                visible_height = min(new_height, zoom_height)
                
                # Create output image with exact base_resized dimensions
                output = np.zeros_like(base_resized)
                
                # Ensure we don't exceed the output dimensions
                visible_height = min(visible_height, output.shape[0])
                visible_width = min(visible_width, output.shape[1])
                
                # Calculate source and destination regions
                src_x = max(0, -x_offset)
                src_y = max(0, -y_offset)
                dst_x = max(0, x_offset)
                dst_y = max(0, y_offset)
                
                # Ensure we don't exceed array bounds
                visible_height = min(visible_height, 
                                   output.shape[0] - dst_y,
                                   zoomed.shape[0] - src_y)
                visible_width = min(visible_width,
                                  output.shape[1] - dst_x,
                                  zoomed.shape[1] - src_x)
                
                # Copy visible portion
                if visible_height > 0 and visible_width > 0:
                    output[dst_y:dst_y+visible_height, dst_x:dst_x+visible_width] = \
                        zoomed[src_y:src_y+visible_height, src_x:src_x+visible_width]
                
                return output
                
            except Exception as e:
                print(f"Zoom error handled: {str(e)}")
                # If any error occurs during zooming, return the base resized image
                return base_resized
        
        return base_resized

    def display_to_original_coords(self, point):
        """Convert coordinates from display image to original image"""
        if point is None:
            return None
        x, y = point
        
        # Calculate the height to maintain aspect ratio (same as in resize_image)
        height, width = self.current_image.shape[:2]
        aspect_ratio = height / width
        new_height = int(self.display_width * aspect_ratio)
        
        # Calculate scale factors
        scale_x = width / self.display_width
        scale_y = height / new_height
        
        # If zoomed, adjust coordinates
        if self.zoom_scale != 1.0:
            # Remove offset
            x = x - self.display_offset[0]
            y = y - self.display_offset[1]
            # Remove zoom
            x = x / self.zoom_scale
            y = y / self.zoom_scale
        
        # Convert to original image coordinates
        original_x = int(x * scale_x)
        original_y = int(y * scale_y)
        
        return (original_x, original_y)

    def original_to_display_coords(self, point):
        """Convert coordinates from original image to display image"""
        if point is None:
            return None
        x, y = point
        
        # First, adjust for the overall image scaling
        x = x / self.scale_factor
        y = y / self.scale_factor
        
        # Then adjust for zoom and offset
        if self.zoom_scale != 1.0:
            # Adjust for zoom
            x = x * self.zoom_scale
            y = y * self.zoom_scale
            # Adjust for offset
            x = x + self.display_offset[0]
            y = y + self.display_offset[1]
        
        return (int(x), int(y))

    def detect_scale_edge(self, image):
        """Detect the scale edge and return the contour"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Canny edge detection with adjusted parameters
            edges = cv2.Canny(gray, 50, 150)
            
            # Apply morphological operations to clean up edges
            kernel = np.ones((3,3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
            
            # Find the largest contour (should be the scale)
            self.scale_contour = max(contours, key=cv2.contourArea)
            
            # Draw the scale contour
            cv2.drawContours(image, [self.scale_contour], -1, (0, 255, 0), 2)
            
            return True  # Changed to return True instead of self.scale_contour
            
        except Exception as e:
            print(f"Error in edge detection: {str(e)}")
            return None

    def is_point_inside_scale(self, point):
        """Check if a point is inside the scale or allow anywhere in manual mode"""
        if self.manual_mode or self.manual_scale_mode:  # Allow clicks anywhere in either manual mode
            return True
        if self.scale_contour is None:
            return False
        return cv2.pointPolygonTest(self.scale_contour, point, False) >= 0

    def find_intersection_with_contour(self, start_point, angle):
        if self.scale_contour is None:
            return None
            
        # Create a line from start point in the given angle
        height, width = self.processed_image.shape[:2]
        radius = max(width, height)
        end_x = int(start_point[0] + radius * np.cos(np.radians(angle)))
        end_y = int(start_point[1] + radius * np.sin(np.radians(angle)))
        
        # Find intersection with contour
        min_dist = float('inf')
        intersection = None
        
        for i in range(len(self.scale_contour)):
            pt1 = self.scale_contour[i][0]
            pt2 = self.scale_contour[(i + 1) % len(self.scale_contour)][0]
            
            # Calculate intersection
            x1, y1 = pt1
            x2, y2 = pt2
            x3, y3 = start_point
            x4, y4 = (end_x, end_y)
            
            denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if denominator == 0:
                continue
                
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
            u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denominator
            
            if 0 <= t <= 1 and 0 <= u <= 1:
                px = int(x1 + t * (x2 - x1))
                py = int(y1 + t * (y2 - y1))
                dist = np.sqrt((px - start_point[0])**2 + (py - start_point[1])**2)
                
                if dist < min_dist:
                    min_dist = dist
                    intersection = (px, py)
        
        return intersection

    def set_center(self, display_point):
        # Convert display coordinates to original image coordinates
        original_point = self.display_to_original_coords(display_point)
        if original_point and self.is_point_inside_scale(original_point):
            self.scale_center = original_point
            # Redraw everything
            self.processed_image = self.current_image.copy()
            
            # Ne dessiner le contour vert que si on n'est pas en mode manuel
            if not self.manual_mode and self.scale_contour is not None:
                cv2.drawContours(self.processed_image, [self.scale_contour], -1, (0, 255, 0), 2)
            
            cv2.circle(self.processed_image, self.scale_center, 5, (0, 0, 255), -1)
            
            # Find intersection with contour and draw line
            if self.manual_mode:
                # En mode manuel, étendre la ligne jusqu'au bord de l'image
                intersection = self.draw_measurement_line(self.processed_image, self.scale_center)
            else:
                # En mode automatique, utiliser l'intersection avec le contour
                intersection = self.find_intersection_with_contour(self.scale_center, self.measurement_angle)
                if intersection:
                    cv2.line(self.processed_image, self.scale_center, intersection, (255, 0, 0), 2)
                    self.measurement_line = intersection
            
            self.redraw_scale()  # Redessiner l'échelle
            self.redraw_interface()
            return True
        return False

    def count_rings_along_line(self):
        if self.scale_center is None or self.measurement_line is None:
            return 0

        # Get the line points
        start = self.scale_center
        end = self.measurement_line
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
        
        # Get the line profile
        line_points = np.linspace(start, end, 100).astype(int)
        profile = [gray[y, x] for x, y in line_points]
        
        # Find peaks in the profile (potential rings)
        peaks = []
        for i in range(1, len(profile)-1):
            if profile[i] > profile[i-1] and profile[i] > profile[i+1]:
                peaks.append(i)
        
        # Draw markers at detected rings
        self.ring_positions = []
        for peak in peaks:
            x, y = line_points[peak]
            cv2.circle(self.processed_image, (x, y), 3, (0, 255, 255), -1)
            self.ring_positions.append((x, y))
        
        return len(peaks)

    def draw_measurement_line(self, image, center, angle=None):
        if angle is not None:
            self.measurement_angle = angle
            
        if self.manual_mode:
            # In manual mode, if we have measurement points, draw line only up to the last point
            if len(self.measurement_points) > 0:
                self.measurement_line = self.measurement_points[-1]  # Set measurement line end to last point
                cv2.line(image, center, self.measurement_line, (255, 0, 0), 2)
                return self.measurement_line
            else:
                # Draw full line only when placing initial points
                height, width = image.shape[:2]
                dx = np.cos(np.radians(self.measurement_angle))
                dy = np.sin(np.radians(self.measurement_angle))
                
                # Calculer les intersections avec les bords
                intersections = []
                
                # Intersection avec les bords verticaux (x = 0 et x = width-1)
                if abs(dx) > 1e-6:  # Éviter la division par zéro
                    t1 = -center[0] / dx  # Pour x = 0
                    t2 = (width-1 - center[0]) / dx  # Pour x = width-1
                    
                    # Vérifier les intersections dans la direction positive
                    if t2 > 0:
                        y = center[1] + t2 * dy
                        if 0 <= y < height:
                            intersections.append((width-1, int(y)))
                    
                    # Vérifier les intersections dans la direction négative
                    if t1 > 0:
                        y = center[1] + t1 * dy
                        if 0 <= y < height:
                            intersections.append((0, int(y)))
                
                # Intersection avec les bords horizontaux (y = 0 et y = height-1)
                if abs(dy) > 1e-6:  # Éviter la division par zéro
                    t3 = -center[1] / dy  # Pour y = 0
                    t4 = (height-1 - center[1]) / dy  # Pour y = height-1
                    
                    # Vérifier les intersections dans la direction positive
                    if t4 > 0:
                        x = center[0] + t4 * dx
                        if 0 <= x < width:
                            intersections.append((int(x), height-1))
                    
                    # Vérifier les intersections dans la direction négative
                    if t3 > 0:
                        x = center[0] + t3 * dx
                        if 0 <= x < width:
                            intersections.append((int(x), 0))
                
                # Trouver le point d'intersection le plus éloigné
                if intersections:
                    end_point = max(intersections, 
                                  key=lambda p: (p[0] - center[0])**2 + (p[1] - center[1])**2)
                    cv2.line(image, center, end_point, (255, 0, 0), 2)
                    self.measurement_line = end_point
                    return end_point
                
                return None
        else:
            # Mode automatique: utiliser l'intersection avec le contour
            intersection = self.find_intersection_with_contour(center, self.measurement_angle)
            if intersection:
                cv2.line(image, center, intersection, (255, 0, 0), 2)
                self.measurement_line = intersection
        
        # Après avoir dessiné la ligne de mesure, toujours redessiner l'échelle
        self.redraw_scale()  # Déplacé à la fin de la méthode
        self.redraw_interface()
        return intersection

    def process_image(self, image_path):
        """Process a new image, attempting automatic detection first"""
        self.current_image_path = image_path
        self.current_image = cv2.imread(image_path)
        if self.current_image is None:
            print(f"Error: Could not read image {image_path}")
            return False
        
        # Create a copy for processing
        self.processed_image = self.current_image.copy()
        
        # Try automatic scale detection first
        scale_detected = self.detect_scale_squares(self.processed_image)
        if not scale_detected:
            print("Automatic scale detection failed.")
            print("Press 's' to switch to manual scale mode.")
        
        # Try edge detection
        edge_detected = self.detect_scale_edge(self.processed_image)
        if edge_detected is None:  # Changed from 'if not edge_detected:'
            print("Edge detection failed.")
            print("Press 'm' to switch to manual edge mode.")
        
        # Always return True to prevent skipping images
        return True

    def detect_scale_squares(self, image):
        """Detect the scale squares by finding the main vertical line and horizontal lines to its left"""
        # Get the right portion of the image (first third)
        height, width = image.shape[:2]
        roi_x = int(width / 4)  # Start from the first third of the image
        roi = image[0:height, 0:roi_x].copy()
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Try different threshold values
        threshold_values = [100, 80, 120, 60, 140]
        kernel_sizes = [(width//4, 1), (width//3, 1), (width//5, 1), (width//2, 1)]  # Added larger kernel
        
        best_scale = None
        best_lines = None
        best_vertical = None
        target_scale = 140  # Target scale in pixels/mm
        scale_tolerance = 30  # Accept scales within ±30 pixels/mm
        
        for threshold in threshold_values:
            # Apply binary threshold to isolate black lines
            _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
            
            # Find vertical lines first
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, height//3))
            vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
            
            # Find contours of vertical lines
            contours, _ = cv2.findContours(vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                continue
            
            # Find the longest vertical line
            main_vertical = None
            max_height = 0
            vertical_x = 0
            
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if h > max_height:
                    max_height = h
                    main_vertical = cnt
                    vertical_x = x
            
            if main_vertical is None:
                continue
            
            # Try different kernel sizes for horizontal lines
            for kernel_size in kernel_sizes:
                # Look for horizontal lines in the entire right portion
                horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
                horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
                
                # Find contours of horizontal lines
                contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Filter and sort horizontal lines
                horizontal_segments = []
                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    if w > width//20:  # Reduced minimum length requirement
                        horizontal_segments.append((y, x, x + w))
                
                if len(horizontal_segments) >= 2:
                    # Sort by y-coordinate
                    horizontal_segments.sort()
                    
                    # Calculate distances between consecutive lines
                    distances = []
                    for i in range(len(horizontal_segments)-1):
                        y1 = horizontal_segments[i][0]
                        y2 = horizontal_segments[i+1][0]
                        distances.append(abs(y2 - y1))
                    
                    if distances:
                        # Calculate potential scale
                        median_dist = np.median(distances)
                        scale = (3 * median_dist) / 10.0  # pixels per mm
                        
                        # Check if this scale is closer to our target
                        if abs(scale - target_scale) < scale_tolerance:
                            if best_scale is None or abs(scale - target_scale) < abs(best_scale - target_scale):
                                best_scale = scale
                                best_lines = horizontal_segments
                                best_vertical = main_vertical
        
        if best_scale is not None:
            self.pixels_per_mm = best_scale
            # Save the detected lines for redrawing
            self.best_vertical = best_vertical
            self.best_lines = best_lines
            
            # Draw detected lines on the original image
            # Draw main vertical line
            x, y, w, h = cv2.boundingRect(best_vertical)
            cv2.line(image, (x, y), (x, y + h), (0, 255, 255), 2)
            
            # Draw horizontal lines
            for y, x1, x2 in best_lines:
                cv2.line(image, (x1, y), (x2, y), (0, 255, 255), 2)
            
            # Draw scale information
            cv2.putText(image, f"Scale: {self.pixels_per_mm:.2f} px/mm", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Save scale points for manual mode
            if best_lines and len(best_lines) >= 2:
                y1, x1, _ = best_lines[0]
                y2, x2, _ = best_lines[1]
                self.scale_points = [(x1, y1), (x2, y2)]
                self.scale_defined = True
            
            print(f"Scale detected: {self.pixels_per_mm:.2f} px/mm")
            return True
        
        return False

    def validate_current_state(self):
        """Validate the current state and move to the next one"""
        if not self.scale_defined:
            print("Please define the scale first (automatic or manual with 's' key)")
            return False
        
        if self.current_state == "center":
            if self.scale_center is None:
                print("Please place the center point first")
                return False
            self.center_validated = True
            self.current_state = "points"
            self.point_validated = True  # Reset point validation state
            print("Center point validated. Now place measurement points (up to 10). Press 'v' to validate each point.")
            return False
        elif self.current_state == "points":
            if len(self.measurement_points) == 0:
                print("Please place a measurement point")
                return False
            self.point_validated = True  # Validate the current point
            print(f"Point {len(self.measurement_points)} validated. Place next point or press 'f' to finish and save.")
            return False
        return True

    def reset_state(self):
        """Reset all states for next image processing"""
        self.scale_center = None
        self.measurement_line = None
        self.measurement_points = []
        self.center_validated = False
        self.point1_validated = False
        self.point2_validated = False
        self.current_state = "center"

    def calculate_distances(self):
        if len(self.measurement_points) == 2:
            # Calculate distances between points
            p1, p2 = self.measurement_points
            d1 = np.sqrt((p1[0] - self.scale_center[0])**2 + (p1[1] - self.scale_center[1])**2)
            d2 = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            d3 = np.sqrt((p2[0] - self.measurement_line[0])**2 + (p2[1] - self.measurement_line[1])**2)
            
            print("\nDistances:")
            print(f"Center to first point: {d1:.2f} pixels")
            print(f"First to second point: {d2:.2f} pixels")
            print(f"Second point to edge: {d3:.2f} pixels")

    def set_measurement_point(self, display_point):
        if not self.center_validated or not self.measurement_line:
            return False
        
        # Convert display coordinates to original image coordinates
        original_point = self.display_to_original_coords(display_point)
        if original_point:
            # Project point onto line
            projected_point = self.project_point_to_line(original_point, self.scale_center, self.measurement_line)
            if projected_point:
                # Check if the projected point is within the line segment bounds with higher tolerance in manual mode
                tolerance = 20 if self.manual_mode else 10
                if self.is_point_on_line(projected_point, self.scale_center, self.measurement_line, tolerance=tolerance):
                    # If we have an unvalidated point, replace it instead of adding a new one
                    if not self.point_validated and len(self.measurement_points) > 0:
                        self.measurement_points[-1] = projected_point
                        print("Point position updated. Press 'v' to validate.")
                    else:
                        # Add new point only if previous point was validated or no points exist
                        if len(self.measurement_points) < self.MAX_POINTS:
                            self.measurement_points.append(projected_point)
                            self.point_validated = False
                            print(f"Point {len(self.measurement_points)} placed. Press 'v' to validate or click again to adjust position.")
                        else:
                            print(f"Maximum number of points ({self.MAX_POINTS}) reached.")
                    
                    # Redraw everything
                    self.redraw_interface()
                    return True
                else:
                    print("Please click closer to the measurement line.")
            else:
                print("Could not project point onto line.")
        return False

    def project_point_to_line(self, point, line_start, line_end):
        """Project a point orthogonally onto a line segment"""
        x, y = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        # Calculate the line vector
        dx = x2 - x1
        dy = y2 - y1
        
        # Calculate the point vector from line start
        px = x - x1
        py = y - y1
        
        # Calculate the dot product
        dot = px * dx + py * dy
        
        # Calculate the squared length of the line
        len_sq = dx * dx + dy * dy
        
        if len_sq == 0:
            return None
        
        # Calculate the parameter of the projection
        param = dot / len_sq
        
        # Calculate the projected point
        proj_x = x1 + param * dx
        proj_y = y1 + param * dy
        
        return (int(proj_x), int(proj_y))

    def is_point_on_line(self, point, line_start, line_end, tolerance=10):  # Increased tolerance
        """Check if a point is close enough to the measurement line"""
        if line_start is None or line_end is None:
            return False
        
        # Calculate distance from point to line
        x, y = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        # Calculate the distance from point to line segment
        A = x - x1
        B = y - y1
        C = x2 - x1
        D = y2 - y1
        
        dot = A * C + B * D
        len_sq = C * C + D * D
        
        if len_sq == 0:
            return False
        
        param = dot / len_sq
        
        # Allow points slightly beyond the line segment ends
        if param < -0.1 or param > 1.1:  # More lenient bounds
            return False
        
        # Calculate projection point
        proj_x = x1 + param * C
        proj_y = y1 + param * D
        
        # Calculate distance from point to projection
        dist = np.sqrt((x - proj_x)**2 + (y - proj_y)**2)
        
        return dist <= tolerance

    def save_to_csv(self, image_name):
        try:
            csv_file = "scale_measurements.csv"
            
            # Create new file with headers if it doesn't exist
            if not os.path.exists(csv_file):
                with open(csv_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Date", "Time", "Image Name", "Total Line Length", 
                                   "Center to First Point", "First to Second Point", "Second Point to Edge"])
            
            # Calculate distances
            p1, p2 = self.measurement_points
            d1 = np.sqrt((p1[0] - self.scale_center[0])**2 + (p1[1] - self.scale_center[1])**2)
            d2 = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            d3 = np.sqrt((p2[0] - self.measurement_line[0])**2 + (p2[1] - self.measurement_line[1])**2)
            
            # Calculate total line length
            total_length = np.sqrt((self.measurement_line[0] - self.scale_center[0])**2 + 
                                 (self.measurement_line[1] - self.scale_center[1])**2)
            
            # Get current date and time
            now = datetime.datetime.now()
            current_date = now.strftime("%Y-%m-%d")
            current_time = now.strftime("%H:%M:%S")
            
            # Append data
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([current_date, current_time, image_name, total_length, d1, d2, d3])
            
            print(f"\nResults saved to {csv_file}")
            print(f"Total line length: {total_length:.2f} pixels")
            
        except Exception as e:
            print(f"Error saving to CSV: {str(e)}")

    def adjust_zoom(self, x, y, factor):
        """Adjust zoom centered on mouse position"""
        old_scale = self.zoom_scale
        self.zoom_scale *= factor
        self.zoom_scale = max(1.0, min(5.0, self.zoom_scale))  # Limit zoom range
        
        if self.zoom_scale != old_scale:
            # Adjust offset to keep zoom centered on mouse position
            scale_change = self.zoom_scale / old_scale
            x_center = x - self.display_offset[0]
            y_center = y - self.display_offset[1]
            
            new_x = int(x_center * scale_change)
            new_y = int(y_center * scale_change)
            
            self.display_offset = (
                self.display_offset[0] - (new_x - x_center),
                self.display_offset[1] - (new_y - y_center)
            )

    def undo_validation(self):
        """Undo the last validation step"""
        if self.current_state == "points" and len(self.measurement_points) > 0:
            self.measurement_points.pop()  # Remove last point
            print(f"Removed last point. {len(self.measurement_points)} points remaining.")
        elif self.center_validated:
            self.current_state = "center"
            self.center_validated = False
            self.scale_center = None
            self.measurement_points = []
            self.measurement_line_angle = 0
            print("Undid center validation. You can now place the center point again.")
        
        # Reset and redraw
        self.processed_image = self.current_image.copy()
        cv2.drawContours(self.processed_image, [self.scale_contour], -1, (0, 255, 0), 2)
        if self.center_validated:
            cv2.circle(self.processed_image, self.scale_center, 5, (0, 0, 255), -1)
            self.draw_measurement_line(self.processed_image, self.scale_center)
        for i, point in enumerate(self.measurement_points):
            color = (0, 255, 0) if i == 0 else (255, 0, 255)
            cv2.circle(self.processed_image, point, 5, color, -1)
        
        self.redraw_scale()  # Redessiner l'échelle
        self.redraw_interface()

    def save_results(self):
        """Save the results to CSV and save the processed image"""
        try:
            # Create the "Processed" and "Treated" folders if they don't exist
            processed_dir = "Processed"
            treated_dir = "Treated"
            for directory in [processed_dir, treated_dir]:
                if not os.path.exists(directory):
                    os.makedirs(directory)
            
            # Save the processed image with measurements
            image_name = os.path.basename(self.current_image_path)
            processed_image_path = os.path.join(processed_dir, f"processed_{image_name}")
            cv2.imwrite(processed_image_path, self.processed_image)
            print(f"\nProcessed image saved to: {processed_image_path}")
            
            # Move original image to Treated folder
            treated_image_path = os.path.join(treated_dir, image_name)
            os.rename(self.current_image_path, treated_image_path)
            print(f"Original image moved to: {treated_image_path}")
            
            # Create CSV file with headers if it doesn't exist
            if not os.path.exists(self.csv_path):
                with open(self.csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    headers = ['Image_Name', 'Fish_ID', 'Scale_Number', 'Date', 'Time', 'Scale_px_per_mm']
                    headers.append('Center_to_First_Point_mm')
                    for i in range(1, self.MAX_POINTS):
                        headers.append(f'Point{i}_to_Point{i+1}_mm')
                    headers.append('Total_Length_mm')
                    writer.writerow(headers)
            
            # Extract Fish ID and Scale Number from image name
            base_name = os.path.splitext(image_name)[0]  # Remove extension
            try:
                fish_id, scale_number = base_name.split('_')
            except ValueError:
                fish_id = base_name
                scale_number = "NA"
            
            # Calculate distances
            distances = []
            
            # First distance: center to first point
            if len(self.measurement_points) > 0:
                dist = np.linalg.norm(np.array(self.measurement_points[0]) - np.array(self.scale_center))
                distances.append(dist / self.pixels_per_mm)  # Convert to mm
            else:
                distances.append(0)
            
            # Distances between consecutive measurement points and to edge in automatic mode
            points_to_process = self.measurement_points.copy()
            if not self.manual_mode and self.measurement_line:
                # In automatic mode, add the edge point as the last measurement point
                points_to_process.append(self.measurement_line)
            
            # Calculate distances between consecutive points
            for i in range(len(points_to_process) - 1):
                dist = np.linalg.norm(
                    np.array(points_to_process[i+1]) - np.array(points_to_process[i]))
                distances.append(dist / self.pixels_per_mm)  # Convert to mm
            
            # Fill remaining point-to-point distances with 0
            while len(distances) < self.MAX_POINTS:
                distances.append(0)
            
            # Total distance
            if self.manual_mode and len(self.measurement_points) > 0:
                # In manual mode, total distance is from center to last measurement point
                total_distance = np.linalg.norm(
                    np.array(self.measurement_points[-1]) - np.array(self.scale_center))
            else:
                # In automatic mode, use full line length
                total_distance = np.linalg.norm(
                    np.array(self.measurement_line) - np.array(self.scale_center))
            distances.append(total_distance / self.pixels_per_mm)  # Convert to mm
            
            # Get current date and time
            now = datetime.datetime.now()
            current_date = now.strftime("%Y-%m-%d")
            current_time = now.strftime("%H:%M:%S")
            
            # Prepare row data with new columns
            row_data = [
                os.path.basename(self.current_image_path),
                fish_id,
                scale_number,
                current_date,
                current_time,
                f"{self.pixels_per_mm:.2f}"  # Add scale value
            ]
            row_data.extend(distances)  # Add all distances
            
            # Save to CSV
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row_data)
            
            # Print summary
            print(f"\nResults saved for Fish ID: {fish_id}, Scale: {scale_number}")
            print(f"Scale: {self.pixels_per_mm:.2f} px/mm")
            print("Distances (in mm):")
            if len(points_to_process) > 0:
                print(f"Center to Point 1: {distances[0]:.2f}")
                for i in range(len(points_to_process) - 1):
                    print(f"Point {i+1} to Point {i+2}: {distances[i+1]:.2f}")
            print(f"Total distance: {distances[-1]:.2f}")
            
        except Exception as e:
            print(f"Error saving results: {str(e)}")

    def redraw_scale(self):
        """Helper method to redraw scale points and text"""
        if len(self.scale_points) == 2:
            # Dessiner les points d'échelle
            for point in self.scale_points:
                cv2.circle(self.processed_image, point, 5, (0, 255, 255), -1)
            # Dessiner la ligne d'échelle
            cv2.line(self.processed_image, self.scale_points[0], self.scale_points[1], (0, 255, 255), 2)
            # Afficher la valeur de l'échelle
            if self.pixels_per_mm is not None:
                cv2.putText(self.processed_image, 
                           f"Scale: {self.pixels_per_mm:.2f} px/mm", 
                           (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           1, 
                           (0, 255, 255), 
                           2)

    def redraw_interface(self):
        """Redraw all interface elements consistently"""
        # Start with a fresh copy of the image
        self.processed_image = self.current_image.copy()
        
        # Draw the image name at the top
        image_name = os.path.basename(self.current_image_path)
        cv2.putText(self.processed_image, image_name, 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw the scale text if we have pixels_per_mm
        if self.pixels_per_mm is not None:
            cv2.putText(self.processed_image, f"Scale: {self.pixels_per_mm:.2f} px/mm", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Draw the detected scale lines in yellow only if not in manual mode
        if not self.manual_scale_mode and hasattr(self, 'best_vertical') and self.best_vertical is not None:
            # Draw main vertical line
            x, y, w, h = cv2.boundingRect(self.best_vertical)
            cv2.line(self.processed_image, (x, y), (x, y + h), (0, 255, 255), 2)
            
            # Draw horizontal lines
            if self.best_lines:
                for y, x1, x2 in self.best_lines:
                    cv2.line(self.processed_image, (x1, y), (x2, y), (0, 255, 255), 2)
        
        # Draw manual scale points if in manual mode
        if self.manual_scale_mode and hasattr(self, 'scale_points') and len(self.scale_points) > 0:
            for point in self.scale_points:
                cv2.circle(self.processed_image, point, 5, (0, 255, 255), -1)
            if len(self.scale_points) == 2:
                cv2.line(self.processed_image, self.scale_points[0], self.scale_points[1], (0, 255, 255), 2)
        
        # Draw the green contour only if not in manual mode
        if not self.manual_mode and self.scale_contour is not None:
            cv2.drawContours(self.processed_image, [self.scale_contour], -1, (0, 255, 0), 2)
        
        # Draw center point and measurement line
        if self.scale_center:
            cv2.circle(self.processed_image, self.scale_center, 5, (0, 0, 255), -1)
            if not self.center_validated:
                if self.manual_mode:
                    self.draw_measurement_line(self.processed_image, self.scale_center)
                else:
                    intersection = self.find_intersection_with_contour(self.scale_center, self.measurement_angle)
                    if intersection:
                        cv2.line(self.processed_image, self.scale_center, intersection, (255, 0, 0), 2)
                        self.measurement_line = intersection
            elif self.measurement_line:
                cv2.line(self.processed_image, self.scale_center, self.measurement_line, (255, 0, 0), 2)
        
        # Draw measurement points
        for i, point in enumerate(self.measurement_points):
            color = (0, 255, 0) if i == 0 else (255, 0, 255)
            cv2.circle(self.processed_image, point, 5, color, -1)

def mouse_callback(event, x, y, flags, param):
    analyzer = param
    if event == cv2.EVENT_LBUTTONDOWN:
        display_coords = (x, y)
        image_coords = analyzer.display_to_original_coords(display_coords)
        
        # Handle manual scale mode - allow clicks anywhere
        if analyzer.manual_scale_mode:
            if len(analyzer.scale_points) < 2:
                analyzer.scale_points.append(image_coords)
                if len(analyzer.scale_points) == 2:
                    # Calculate pixels per mm based on 3.33mm distance
                    dist = np.sqrt((analyzer.scale_points[1][0] - analyzer.scale_points[0][0])**2 + 
                                 (analyzer.scale_points[1][1] - analyzer.scale_points[0][1])**2)
                    analyzer.pixels_per_mm = dist / 3.33
                    analyzer.scale_defined = True
                    print(f"Scale set manually: {analyzer.pixels_per_mm:.2f} px/mm")
                else:
                    print("First point set. Click second point (3.33mm from first point)")
                analyzer.redraw_interface()
                return
        
        # For center point placement
        if not analyzer.center_validated:
            if analyzer.is_point_inside_scale(image_coords):  # This will now work in manual mode
                analyzer.scale_center = image_coords
                analyzer.redraw_interface()
                print("Center point set. Press 'v' to validate.")
            else:
                print("Please click inside the scale.")
        # For measurement point placement
        elif analyzer.center_validated:
            analyzer.set_measurement_point(display_coords)
        
        analyzer.redraw_interface()
    elif event == cv2.EVENT_MOUSEWHEEL:
        # Windows scroll
        factor = 1.1 if flags > 0 else 0.9
        analyzer.adjust_zoom(x, y, factor)
    elif event == cv2.EVENT_MOUSEHWHEEL:
        # Linux/Mac scroll
        factor = 0.9 if flags > 0 else 1.1
        analyzer.adjust_zoom(x, y, factor)
        
    # Update display
    display_processed = analyzer.resize_image(analyzer.processed_image)
    cv2.imshow("Scale Analysis", display_processed)

def main():
    # Create the test images folder if it doesn't exist
    folder_path = "Test images"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created folder: {folder_path}")
        print("Please add some images to this folder and run the program again.")
        return

    # Get list of image files and sort them alphabetically
    image_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_files.append(os.path.join(root, file))
    
    # Sort files alphabetically
    image_files.sort()
    
    if not image_files:
        print("\nNo images found in the Test images folder or its subfolders.")
        print("Please add some images and run the program again.")
        return

    # Initialize scale analyzer
    analyzer = ScaleAnalyzer(image_files[0], "Output images", "scale_measurements.csv")
    current_image_index = 0
    
    # Set up mouse callback
    cv2.namedWindow("Scale Analysis")
    cv2.setMouseCallback("Scale Analysis", mouse_callback, analyzer)
    
    # Update the image files list after moving a file
    def update_image_files():
        updated_files = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    updated_files.append(os.path.join(root, file))
        # Sort the updated files list
        updated_files.sort()
        return updated_files
    
    while current_image_index < len(image_files):
        image_path = image_files[current_image_index]
        print(f"\nProcessing image: {image_path}")
        
        if analyzer.process_image(image_path) or analyzer.manual_mode:
            # Resize image for display
            display_processed = analyzer.resize_image(analyzer.processed_image)
            cv2.imshow("Scale Analysis", display_processed)
            
            # Add instructions
            print("\nInstructions:")
            if not analyzer.manual_mode:
                print("1. The green outline shows the detected scale edge")
            print("2. Click inside the scale to set the center point (red dot)")
            print("3. Use 'a'/'d' keys to adjust the line angle")
            print("4. Press 'v' to validate current point and move to next step")
            print("5. Click on the line to place measurement points (green/magenta dots)")
            print("6. Press 'v' to validate each point before placing the next one")
            
            print("\nControls:")
            print("- Mouse wheel: Zoom in/out")
            print("- 'a'/'d' keys: Adjust measurement line angle (left/right)")
            print("- 'v' key: Validate current point")
            print("- 'f' key: Finish and save measurements")
            print("- 'u' key: Undo last point/validation")
            print("- 'm' key: Toggle manual mode (disable edge detection)")
            print("- 's' key: Switch to manual scale mode (define scale manually)")
            print("- 'b' key: Go to previous image")
            print("- 'n' key: Go to next image")
            print("- Press 'q' to quit")
            
            print("\nScale modes:")
            print("- Automatic: Program detects scale marks automatically")
            print("- Manual ('s'): Click two points 3.33mm apart to set scale")
            print("- Edge detection ('m'): Toggle between automatic/manual edge detection")
            
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    return
                elif key == ord('v'):  # Validate key
                    if analyzer.validate_current_state():
                        # If validation returns True (final review validated), move to next image
                        current_image_index += 1
                        break
                elif key == ord('u'):  # Undo key
                    analyzer.undo_validation()
                elif key == ord('b'):  # Previous image
                    if current_image_index > 0:
                        current_image_index -= 1
                        # Reset analyzer with new image
                        analyzer = ScaleAnalyzer(image_files[current_image_index], "Output images", "scale_measurements.csv")
                        analyzer.process_image(image_files[current_image_index])
                        analyzer.point_validated = True
                        analyzer.current_state = "center"
                        analyzer.center_validated = False
                        # Reset mouse callback with new analyzer
                        cv2.setMouseCallback("Scale Analysis", mouse_callback, analyzer)
                        break
                elif key == ord('n'):  # Next image
                    if current_image_index < len(image_files) - 1:
                        current_image_index += 1
                        # Reset analyzer with new image
                        analyzer = ScaleAnalyzer(image_files[current_image_index], "Output images", "scale_measurements.csv")
                        # Conserver les modes manuels mais réinitialiser les points
                        analyzer.manual_mode = analyzer.manual_mode
                        analyzer.manual_scale_mode = analyzer.manual_scale_mode
                        analyzer.process_image(image_files[current_image_index])
                        analyzer.point_validated = True
                        analyzer.current_state = "center"
                        analyzer.center_validated = False
                        # Reset mouse callback with new analyzer
                        cv2.setMouseCallback("Scale Analysis", mouse_callback, analyzer)
                        break
                elif key == ord('d'):  # Right rotation
                    if analyzer.scale_center and not analyzer.center_validated:
                        print(f"Rotating right. Current angle: {analyzer.measurement_angle}")
                        analyzer.measurement_angle += analyzer.angle_step
                        analyzer.redraw_interface()
                        print(f"New angle: {analyzer.measurement_angle}")
                elif key == ord('a'):  # Left rotation
                    if analyzer.scale_center and not analyzer.center_validated:
                        print(f"Rotating left. Current angle: {analyzer.measurement_angle}")
                        analyzer.measurement_angle -= analyzer.angle_step
                        analyzer.redraw_interface()
                        print(f"New angle: {analyzer.measurement_angle}")
                elif key == ord('f'):  # Finish placing points
                    if analyzer.current_state == "points" and len(analyzer.measurement_points) > 0:
                        if analyzer.point_validated:  # Only allow finish if current point is validated
                            analyzer.save_results()  # Save and move the file
                            # Update image files list after moving the file
                            image_files = update_image_files()
                            if not image_files:  # If no more images to process
                                print("\nAll images processed!")
                                cv2.destroyAllWindows()
                                return
                            # Adjust current_image_index if needed
                            if current_image_index >= len(image_files):
                                current_image_index = len(image_files) - 1
                            # Create new analyzer instance for next image
                            analyzer = ScaleAnalyzer(image_files[current_image_index], "Output images", "scale_measurements.csv")
                            analyzer.process_image(image_files[current_image_index])
                            analyzer.point_validated = True
                            analyzer.current_state = "center"
                            analyzer.center_validated = False
                            cv2.setMouseCallback("Scale Analysis", mouse_callback, analyzer)
                            break
                        else:
                            print("Please validate current point first with 'v'")
                elif key == ord('m'):  # Toggle manual mode
                    analyzer.manual_mode = not analyzer.manual_mode
                    if analyzer.manual_mode:
                        print("\nSwitched to manual mode. Edge detection disabled.")
                        # Reset the image without edge detection
                        analyzer.processed_image = analyzer.current_image.copy()
                        analyzer.scale_contour = None  # Clear the contour in manual mode
                        
                        # Redraw existing elements
                        analyzer.redraw_interface()
                    else:
                        print("\nSwitched back to automatic mode.")
                        # Save current scale and points
                        old_scale_points = analyzer.scale_points
                        old_pixels_per_mm = analyzer.pixels_per_mm
                        old_scale_defined = analyzer.scale_defined
                        old_scale_center = analyzer.scale_center
                        old_measurement_points = analyzer.measurement_points
                        
                        # Reload image and edge detection
                        analyzer.process_image(image_path)
                        
                        # Restore previous measurements
                        if old_scale_defined:
                            analyzer.scale_points = old_scale_points
                            analyzer.pixels_per_mm = old_pixels_per_mm
                            analyzer.scale_defined = old_scale_defined
                        if old_scale_center:
                            analyzer.scale_center = old_scale_center
                        if old_measurement_points:
                            analyzer.measurement_points = old_measurement_points
                        
                        analyzer.redraw_interface()
                    
                    # Update display
                    display_processed = analyzer.resize_image(analyzer.processed_image)
                    cv2.imshow("Scale Analysis", display_processed)
                elif key == ord('s'):  # Switch to manual scale mode
                    analyzer.manual_scale_mode = True
                    analyzer.best_vertical = None  # Clear automatic detection
                    analyzer.best_lines = None
                    analyzer.scale_points = []  # Reset scale points
                    analyzer.pixels_per_mm = None  # Reset scale
                    analyzer.scale_defined = False
                    print("Switched to manual scale mode. Click two points on the scale (3.33mm apart).")
                    analyzer.redraw_interface()
                
                # Update display
                display_processed = analyzer.resize_image(analyzer.processed_image)
                cv2.imshow("Scale Analysis", display_processed)
        
        else:
            current_image_index += 1
    
    print("\nAll images processed!")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

