#!/usr/bin/env python3
"""
Thermal Hotpoint Detector and Annotator

This script analyzes thermal images to detect hot regions and cluster them into 
rectangular annotations. It processes faulty images from the dataset and 
identifies areas with elevated temperatures.

"""

import cv2
import numpy as np
import os
import glob
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import namedtuple
import json

# Define a structure for bounding boxes
BoundingBox = namedtuple('BoundingBox', ['x', 'y', 'width', 'height', 'confidence'])

class ThermalHotpointDetector:
    def __init__(self, temperature_threshold=200, min_cluster_size=10, cluster_epsilon=15):
        """
        Initialize the thermal hotpoint detector.
        
        Args:
            temperature_threshold (int): Absolute temperature threshold (0-255 for grayscale, pixels above this are considered hot)
            min_cluster_size (int): Minimum number of pixels to form a cluster
            cluster_epsilon (int): Maximum distance between samples in a cluster
        """
        self.temp_threshold = temperature_threshold
        self.min_cluster_size = min_cluster_size
        self.cluster_epsilon = cluster_epsilon
        
    def remove_white_regions(self, image_rgb):
        """
        Remove white regions from the image using dilation.
        
        Args:
            image_rgb (numpy.ndarray): RGB image
            
        Returns:
            numpy.ndarray: Mask of valid (non-white) regions
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        
        # Detect white regions (threshold at 240 to catch near-white pixels)
        _, white_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        
        # Dilate white regions to remove them more aggressively
        kernel = np.ones((5, 5), np.uint8)
        white_mask_dilated = cv2.dilate(white_mask, kernel, iterations=2)
        
        # Invert to get valid region mask (non-white areas)
        valid_mask = cv2.bitwise_not(white_mask_dilated)
        
        return valid_mask.astype(bool)
    
    def crop_border(self, image, border_percent=10):
        """
        Remove border region from image (10% from each side).
        
        Args:
            image (numpy.ndarray): Input image
            border_percent (int): Percentage of border to remove from each side
            
        Returns:
            tuple: (cropped_image, crop_info) where crop_info contains the crop boundaries
        """
        height, width = image.shape[:2]
        
        # Calculate border size
        border_h = int(height * border_percent / 100)
        border_w = int(width * border_percent / 100)
        
        # Crop the image
        cropped = image[border_h:height-border_h, border_w:width-border_w]
        
        crop_info = {
            'top': border_h,
            'bottom': height - border_h,
            'left': border_w,
            'right': width - border_w
        }
        
        return cropped, crop_info
    
    def load_thermal_image(self, image_path):
        """
        Load a thermal image and extract red channel for thermal analysis.
        Includes preprocessing: border removal and white region masking.
        
        Args:
            image_path (str): Path to the thermal image
            
        Returns:
            tuple: (original_image, red_channel_image, valid_mask, crop_info)
        """
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert to RGB for matplotlib compatibility
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Store original for visualization
        original_image = image_rgb.copy()
        
        # Step 1: Remove 10% border
        cropped_image, crop_info = self.crop_border(image_rgb, border_percent=10)
        
        # Step 2: Remove white regions
        valid_mask = self.remove_white_regions(cropped_image)
        
        # Extract RED channel only for thermal analysis
        # In thermal images: Red = hot (danger), Yellow/Blue = not dangerous
        red_channel = cropped_image[:, :, 0]  # Extract red channel from RGB
        
        # Apply white region mask to red channel
        red_channel_masked = red_channel.copy()
        red_channel_masked[~valid_mask] = 0  # Set white regions to 0
        
        return original_image, red_channel_masked, valid_mask, crop_info, cropped_image
    
    def detect_hot_regions(self, thermal_image):
        """
        Detect hot regions in the thermal image using constant threshold on red channel.
        
        Args:
            thermal_image (numpy.ndarray): Red channel of thermal image
            
        Returns:
            numpy.ndarray: Binary mask of hot regions
        """
        # Use constant threshold - only red pixels above this value are considered hot
        # This focuses on red regions (thermal issues) and ignores yellow/blue regions
        hot_mask = thermal_image > self.temp_threshold
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        hot_mask = cv2.morphologyEx(hot_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        hot_mask = cv2.morphologyEx(hot_mask, cv2.MORPH_CLOSE, kernel)
        
        return hot_mask.astype(bool)
    
    def cluster_hot_points(self, hot_mask):
        """
        Cluster hot points using DBSCAN algorithm.
        
        Args:
            hot_mask (numpy.ndarray): Binary mask of hot regions
            
        Returns:
            list: List of clustered point groups
        """
        # Find coordinates of hot pixels
        hot_points = np.column_stack(np.where(hot_mask))
        
        if len(hot_points) == 0:
            return []
        
        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=self.cluster_epsilon, min_samples=self.min_cluster_size)
        cluster_labels = clustering.fit_predict(hot_points)
        
        # Group points by cluster
        clusters = []
        unique_labels = set(cluster_labels)
        
        for label in unique_labels:
            if label == -1:  # Skip noise points
                continue
            
            cluster_points = hot_points[cluster_labels == label]
            clusters.append(cluster_points)
        
        return clusters
    
    def merge_overlapping_boxes(self, bounding_boxes, iou_threshold=0.3):
        """
        Merge overlapping bounding boxes to avoid duplicate annotations.
        
        Args:
            bounding_boxes (list): List of BoundingBox objects
            iou_threshold (float): IoU threshold for merging (0.3 = 30% overlap)
            
        Returns:
            list: List of non-overlapping BoundingBox objects
        """
        if len(bounding_boxes) == 0:
            return []
        
        # Convert to list of [x1, y1, x2, y2, confidence]
        boxes = []
        for bbox in bounding_boxes:
            x1, y1 = bbox.x, bbox.y
            x2, y2 = bbox.x + bbox.width, bbox.y + bbox.height
            boxes.append([x1, y1, x2, y2, bbox.confidence])
        
        boxes = np.array(boxes)
        merged = []
        used = set()
        
        for i in range(len(boxes)):
            if i in used:
                continue
                
            # Start with current box
            current_box = boxes[i].copy()
            merged_indices = [i]
            
            # Find overlapping boxes
            for j in range(i + 1, len(boxes)):
                if j in used:
                    continue
                
                # Calculate IoU
                x1_i, y1_i, x2_i, y2_i = current_box[:4]
                x1_j, y1_j, x2_j, y2_j = boxes[j][:4]
                
                # Intersection
                x1_inter = max(x1_i, x1_j)
                y1_inter = max(y1_i, y1_j)
                x2_inter = min(x2_i, x2_j)
                y2_inter = min(y2_i, y2_j)
                
                if x1_inter < x2_inter and y1_inter < y2_inter:
                    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
                    
                    # Union
                    area_i = (x2_i - x1_i) * (y2_i - y1_i)
                    area_j = (x2_j - x1_j) * (y2_j - y1_j)
                    union_area = area_i + area_j - inter_area
                    
                    iou = inter_area / union_area if union_area > 0 else 0
                    
                    # If overlapping, merge
                    if iou > iou_threshold:
                        merged_indices.append(j)
                        # Expand current box to include this box
                        current_box[0] = min(current_box[0], boxes[j][0])
                        current_box[1] = min(current_box[1], boxes[j][1])
                        current_box[2] = max(current_box[2], boxes[j][2])
                        current_box[3] = max(current_box[3], boxes[j][3])
                        # Take max confidence
                        current_box[4] = max(current_box[4], boxes[j][4])
            
            # Mark all merged boxes as used
            used.update(merged_indices)
            
            # Convert back to BoundingBox
            x1, y1, x2, y2, conf = current_box
            merged_bbox = BoundingBox(
                x=int(x1),
                y=int(y1),
                width=int(x2 - x1),
                height=int(y2 - y1),
                confidence=float(conf)
            )
            merged.append(merged_bbox)
        
        return merged
    
    def filter_large_boxes(self, bounding_boxes, image_shape, max_area_percent=80):
        """
        Filter out bounding boxes that are too large (>80% of image area).
        
        Args:
            bounding_boxes (list): List of BoundingBox objects
            image_shape (tuple): Shape of the image (height, width)
            max_area_percent (int): Maximum percentage of image area a box can occupy
            
        Returns:
            list: Filtered list of BoundingBox objects
        """
        if len(bounding_boxes) == 0:
            return []
        
        image_area = image_shape[0] * image_shape[1]
        max_area = image_area * (max_area_percent / 100)
        
        filtered_boxes = []
        for bbox in bounding_boxes:
            bbox_area = bbox.width * bbox.height
            if bbox_area <= max_area:
                filtered_boxes.append(bbox)
            else:
                print(f"  → Filtered out large box: {bbox_area}/{image_area} = {(bbox_area/image_area)*100:.1f}% of image")
        
        return filtered_boxes
    
    def create_bounding_boxes(self, clusters, image_shape):
        """
        Create bounding boxes around clustered hot regions.
        Filters out boxes larger than 80% of image area.
        
        Args:
            clusters (list): List of clustered point groups
            image_shape (tuple): Shape of the image (height, width)
            
        Returns:
            list: List of BoundingBox objects
        """
        bounding_boxes = []
        
        for i, cluster in enumerate(clusters):
            if len(cluster) < self.min_cluster_size:
                continue
                
            # Calculate bounding box coordinates
            min_row, min_col = np.min(cluster, axis=0)
            max_row, max_col = np.max(cluster, axis=0)
            
            # Add some padding around the bounding box
            padding = 5
            min_row = max(0, min_row - padding)
            min_col = max(0, min_col - padding)
            
            width = max_col - min_col + 2 * padding
            height = max_row - min_row + 2 * padding
            
            # Calculate confidence based on cluster density
            area = width * height
            density = len(cluster) / area if area > 0 else 0
            confidence = min(1.0, density / 0.1)  # Normalize density to confidence
            
            bbox = BoundingBox(
                x=int(min_col),
                y=int(min_row),
                width=int(width),
                height=int(height),
                confidence=float(confidence)
            )
            bounding_boxes.append(bbox)
        
        # Merge overlapping boxes
        bounding_boxes = self.merge_overlapping_boxes(bounding_boxes)
        
        # Filter out boxes that are too large (>80% of image)
        bounding_boxes = self.filter_large_boxes(bounding_boxes, image_shape, max_area_percent=80)
        
        return bounding_boxes
    
    def annotate_image(self, image, bounding_boxes, crop_info, show_confidence=True):
        """
        Annotate the image with bounding boxes around hot regions.
        Adjusts bounding box coordinates to match original image after cropping.
        
        Args:
            image (numpy.ndarray): Original RGB image (uncropped)
            bounding_boxes (list): List of BoundingBox objects (in cropped coordinates)
            crop_info (dict): Information about how the image was cropped
            show_confidence (bool): Whether to display confidence scores
            
        Returns:
            matplotlib.figure.Figure: Annotated image figure
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)
        ax.set_title('Thermal Image with Hot Region Detection', fontsize=14, fontweight='bold')
        
        # Draw bounding boxes - adjust coordinates back to original image space
        for i, bbox in enumerate(bounding_boxes):
            # Adjust coordinates to account for border cropping
            adjusted_x = bbox.x + crop_info['left']
            adjusted_y = bbox.y + crop_info['top']
            
            # Color code by confidence: red for high confidence, yellow for low
            color = 'red' if bbox.confidence > 0.7 else 'yellow' if bbox.confidence > 0.4 else 'orange'
            
            # Create rectangle patch
            rect = patches.Rectangle(
                (adjusted_x, adjusted_y), bbox.width, bbox.height,
                linewidth=2, edgecolor=color, facecolor='none',
                linestyle='-'
            )
            ax.add_patch(rect)
            
            # Add confidence label if requested
            if show_confidence:
                ax.text(
                    adjusted_x, adjusted_y - 5,
                    f'Hot Region {i+1}\nConf: {bbox.confidence:.2f}',
                    fontsize=8, color=color, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
                )
        
        ax.axis('off')
        plt.tight_layout()
        
        return fig
    
    def process_single_image(self, image_path, output_dir=None, save_annotations=True):
        """
        Process a single thermal image and detect hot regions using red channel.
        Includes preprocessing: border removal, white region masking, and large box filtering.
        
        Args:
            image_path (str): Path to the thermal image
            output_dir (str): Directory to save annotated images
            save_annotations (bool): Whether to save annotation data
            
        Returns:
            dict: Processing results containing bounding boxes and statistics
        """
        print(f"Processing: {os.path.basename(image_path)}")
        
        # Load image, extract red channel, remove border and white regions
        original_image, red_channel_masked, valid_mask, crop_info, cropped_image = self.load_thermal_image(image_path)
        
        # Detect hot regions from red channel only
        hot_mask = self.detect_hot_regions(red_channel_masked)
        
        # Cluster hot points
        clusters = self.cluster_hot_points(hot_mask)
        
        # Create bounding boxes (with large box filtering)
        bounding_boxes = self.create_bounding_boxes(clusters, red_channel_masked.shape)
        
        # Create annotated visualization (using original image)
        fig = self.annotate_image(original_image, bounding_boxes, crop_info)
        
        # Save results if output directory is specified
        if output_dir and save_annotations:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save annotated image
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}_annotated.png")
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            
            # Save annotation data (adjust coordinates back to original image space)
            annotation_data = {
                'image_path': image_path,
                'num_hot_regions': len(bounding_boxes),
                'bounding_boxes': [
                    {
                        'x': int(bbox.x + crop_info['left']),  # Adjust for cropping
                        'y': int(bbox.y + crop_info['top']),   # Adjust for cropping
                        'width': int(bbox.width),
                        'height': int(bbox.height),
                        'confidence': float(bbox.confidence)
                    }
                    for bbox in bounding_boxes
                ],
                'statistics': {
                    'total_hot_pixels': int(np.sum(hot_mask)),
                    'image_dimensions': [int(dim) for dim in original_image.shape[:2]],
                    'cropped_dimensions': [int(dim) for dim in red_channel_masked.shape],
                    'temperature_threshold': self.temp_threshold,
                    'analysis_channel': 'red',
                    'preprocessing': {
                        'border_removed': '10%',
                        'white_regions_masked': True,
                        'large_boxes_filtered': '>80% area'
                    }
                }
            }
            
            json_path = os.path.join(output_dir, f"{base_name}_annotations.json")
            with open(json_path, 'w') as f:
                json.dump(annotation_data, f, indent=2)
            
            print(f"  ✓ Saved annotated image and JSON to {output_dir}")
        
        # Close figure to prevent memory issues
        plt.close(fig)
        
        # Return results
        results = {
            'bounding_boxes': bounding_boxes,
            'num_hot_regions': len(bounding_boxes),
            'total_hot_pixels': int(np.sum(hot_mask)),
            'clusters': clusters,
            'hot_mask': hot_mask
        }
        
        return results
    
    def process_dataset(self, dataset_path, output_dir="output_annotations"):
        """
        Process all faulty images in the dataset.
        
        Args:
            dataset_path (str): Path to the dataset directory
            output_dir (str): Directory to save all annotations
        """
        # Find all faulty image directories
        faulty_patterns = [
            os.path.join(dataset_path, "*/faulty/*.jpg"),
            os.path.join(dataset_path, "*/faulty/*.png")
        ]
        
        all_faulty_images = []
        for pattern in faulty_patterns:
            all_faulty_images.extend(glob.glob(pattern))
        
        if not all_faulty_images:
            print(f"No faulty images found in {dataset_path}")
            return
        
        print(f"Found {len(all_faulty_images)} faulty images to process")
        
        # Process each image
        all_results = []
        images_with_hotspots = 0
        for idx, image_path in enumerate(all_faulty_images, 1):
            try:
                print(f"[{idx}/{len(all_faulty_images)}] Processing: {os.path.basename(image_path)}")
                results = self.process_single_image(image_path, output_dir, save_annotations=False)
                results['image_path'] = image_path
                all_results.append(results)
                
                if results['num_hot_regions'] > 0:
                    images_with_hotspots += 1
                    print(f"  → Found {results['num_hot_regions']} hot region(s)")
                else:
                    print(f"  → No thermal issues detected")
                    
            except Exception as e:
                print(f"  ✗ Error processing {image_path}: {str(e)}")
        
        # Save summary statistics
        if output_dir and all_results:
            summary = {
                'total_images_processed': len(all_results),
                'images_with_thermal_issues': images_with_hotspots,
                'images_without_thermal_issues': len(all_results) - images_with_hotspots,
                'total_hot_regions_detected': sum(r['num_hot_regions'] for r in all_results),
                'average_hot_regions_per_image': float(np.mean([r['num_hot_regions'] for r in all_results])),
                'temperature_threshold_used': self.temp_threshold
            }
            
            summary_path = os.path.join(output_dir, "processing_summary.json")
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"\n{'='*60}")
            print(f"Processing Complete!")
            print(f"{'='*60}")
            print(f"Total images processed: {summary['total_images_processed']}")
            print(f"Images WITH thermal issues: {summary['images_with_thermal_issues']}")
            print(f"Images WITHOUT thermal issues: {summary['images_without_thermal_issues']}")
            print(f"Total hot regions detected: {summary['total_hot_regions_detected']}")
            print(f"Temperature threshold: {summary['temperature_threshold_used']}")
            print(f"\nResults saved to: {output_dir}")
            print(f"{'='*60}")


def main():
    """
    Main function to demonstrate usage of the ThermalHotpointDetector.
    Analyzes RED channel only - Red = thermal danger, Yellow/Blue = safe
    """
    # Initialize detector with custom parameters
    detector = ThermalHotpointDetector(
        temperature_threshold=200,  # Red channel threshold (0-255), only red pixels above this are considered hot
        min_cluster_size=15,        # Minimum 15 pixels per cluster
        cluster_epsilon=20          # Maximum 20 pixel distance for clustering
    )
    
    # Define paths
    dataset_path = "Dataset"
    output_dir = "thermal_annotations"
    
    # Process the entire dataset
    print("Starting thermal hotpoint detection on faulty images...")
    print("Analyzing RED channel only (Red = thermal danger, Yellow/Blue = safe)")
    detector.process_dataset(dataset_path, output_dir)
    
    # Example: Process a single image
    # single_image_path = "Dataset/T1/faulty/T1_faulty_001.jpg"
    # if os.path.exists(single_image_path):
    #     print(f"\nProcessing single image example: {single_image_path}")
    #     results = detector.process_single_image(single_image_path, output_dir)
    #     print(f"Detected {results['num_hot_regions']} hot regions")


if __name__ == "__main__":
    main()