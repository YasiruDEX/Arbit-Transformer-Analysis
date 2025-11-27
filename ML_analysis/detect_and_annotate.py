"""
Single script to detect anomalies and annotate them with bounding boxes and labels
Usage: python detect_and_annotate.py <image_path> --threshold 0.5
"""
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import AnomalyAutoEncoder


def load_model(model_path, device='cpu'):
    """Load trained model"""
    model = AnomalyAutoEncoder(in_channels=3, latent_dim=128)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


def preprocess_image(image_path, img_size=256, crop_border_percent=10):
    """
    Load and preprocess image with border cropping
    
    Args:
        image_path: Path to image
        img_size: Target size for model input
        crop_border_percent: Percentage of border to crop (default: 10%)
    
    Returns:
        image_tensor: Preprocessed tensor for model
        original_image: Full original image (BGR)
        cropped_image_rgb: Cropped image (RGB)
        original_size: Original image size (H, W)
        crop_coords: Crop coordinates (y1, y2, x1, x2)
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    original_size = image.shape[:2]
    h, w = original_size
    
    # Calculate crop coordinates (remove 10% border on all sides)
    crop_percent = crop_border_percent / 100.0
    y1 = int(h * crop_percent)
    y2 = int(h * (1 - crop_percent))
    x1 = int(w * crop_percent)
    x2 = int(w * (1 - crop_percent))
    
    # Crop the image
    cropped_image = image[y1:y2, x1:x2]
    cropped_image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
    
    # Transform for model
    transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    transformed = transform(image=cropped_image_rgb)
    image_tensor = transformed['image'].unsqueeze(0)
    
    crop_coords = (y1, y2, x1, x2)
    
    return image_tensor, image, cropped_image_rgb, original_size, crop_coords


def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalize tensor for visualization"""
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def generate_anomaly_mask(anomaly_map, threshold=0.5):
    """Generate binary segmentation mask"""
    # Normalize to [0, 1]
    anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
    
    # Apply threshold
    mask = (anomaly_map > threshold).astype(np.uint8) * 255
    
    # Morphological operations to clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask


def calculate_blue_percentage(image_bgr, x, y, w, h):
    """
    Calculate percentage of blue pixels in a bounding box region
    
    Args:
        image_bgr: Original image in BGR format
        x, y, w, h: Bounding box coordinates
    
    Returns:
        Percentage of blue pixels (0-100)
    """
    # Extract ROI
    roi = image_bgr[y:y+h, x:x+w]
    
    # Convert to HSV for better blue detection
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Define blue color range in HSV
    # Blue hue is roughly 100-130 in OpenCV's 0-180 scale
    lower_blue = np.array([90, 50, 50])    # Lower bound: hue, saturation, value
    upper_blue = np.array([130, 255, 255])  # Upper bound
    
    # Create mask for blue pixels
    blue_mask = cv2.inRange(roi_hsv, lower_blue, upper_blue)
    
    # Calculate percentage
    total_pixels = w * h
    blue_pixels = np.count_nonzero(blue_mask)
    blue_percentage = (blue_pixels / total_pixels) * 100
    
    return blue_percentage


def find_contours_and_draw_boxes(image, mask, original_size, crop_coords, min_area=100, max_area=None, max_annotations=3, blue_threshold=None):
    """
    Find contours in mask and draw bounding boxes with labels
    Maps detections from cropped region back to full original image
    
    Args:
        image: Full original image (BGR format)
        mask: Binary mask (256x256) from cropped region
        original_size: Original image size (H, W)
        crop_coords: Crop coordinates (y1, y2, x1, x2)
        min_area: Minimum contour area to consider
        max_area: Maximum contour area to consider (None = no limit)
        max_annotations: Maximum number of annotations to show (top N by confidence)
        blue_threshold: Maximum percentage of blue pixels allowed (None = no filtering)
    
    Returns:
        Annotated image, list of bounding boxes
    """
    y1, y2, x1, x2 = crop_coords
    cropped_h = y2 - y1
    cropped_w = x2 - x1
    
    # Resize mask to cropped image size
    mask_resized = cv2.resize(mask, (cropped_w, cropped_h), interpolation=cv2.INTER_NEAREST)
    
    # Find contours
    contours, hierarchy = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create annotated image
    annotated = image.copy()
    
    # Filter and collect all valid detections
    valid_boxes = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filter by area
        if area < min_area:
            continue
        
        if max_area is not None and area > max_area:
            continue
        
        # Get bounding box in cropped coordinates
        x_crop, y_crop, w, h = cv2.boundingRect(contour)
        
        # Map to original image coordinates
        x_orig = x_crop + x1
        y_orig = y_crop + y1
        
        # Calculate anomaly score (percentage of pixels in bounding box that are anomalous)
        roi_mask = mask_resized[y_crop:y_crop+h, x_crop:x_crop+w]
        anomaly_percentage = (np.sum(roi_mask > 0) / (w * h)) * 100
        
        # Check blue percentage if threshold is set
        if blue_threshold is not None:
            blue_pct = calculate_blue_percentage(image, x_orig, y_orig, w, h)
            if blue_pct > blue_threshold:
                # Skip this detection due to excessive blue
                continue
        else:
            blue_pct = 0.0
        
        valid_boxes.append({
            'bbox': (x_orig, y_orig, w, h),  # Store in original coordinates
            'area': area,
            'score': anomaly_percentage,
            'contour': contour,
            'blue_pct': blue_pct if blue_threshold is not None else None
        })
    
    # Sort by confidence score (descending) and take top N
    valid_boxes.sort(key=lambda x: x['score'], reverse=True)
    top_boxes = valid_boxes[:max_annotations]
    
    # Assign IDs to top boxes
    for idx, box in enumerate(top_boxes):
        box['id'] = idx + 1
    
    # Draw the top N annotations with bounding boxes and labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    
    for box in top_boxes:
        x_orig, y_orig, w, h = box['bbox']
        
        # Draw red bounding box
        cv2.rectangle(annotated, (x_orig, y_orig), (x_orig + w, y_orig + h), 
                     (0, 0, 255), 2)
        
        # Draw blue contour outline (map contour to original coordinates)
        contour_mapped = box['contour'].copy()
        contour_mapped[:, 0, 0] += x1  # Add x offset
        contour_mapped[:, 0, 1] += y1  # Add y offset
        cv2.drawContours(annotated, [contour_mapped], -1, (255, 0, 0), 2)
        
        # Prepare label
        label = f"Anomaly #{box['id']}"
        score_text = f"{box['score']:.1f}%"
        
        # Calculate label background size
        (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        (score_w, score_h), _ = cv2.getTextSize(score_text, font, font_scale, thickness)
        
        # Calculate label position (center top of anomaly region)
        label_x = x_orig + (w // 2) - (max(label_w, score_w) // 2)
        label_y = y_orig - 10
        
        # Ensure label stays within image bounds
        label_x = max(5, min(label_x, annotated.shape[1] - max(label_w, score_w) - 10))
        label_y = max(label_h + score_h + 20, label_y)
        
        # Draw label background (semi-transparent red)
        overlay = annotated.copy()
        cv2.rectangle(overlay, 
                     (label_x - 5, label_y - label_h - score_h - 15), 
                     (label_x + max(label_w, score_w) + 10, label_y + 5), 
                     (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.8, annotated, 0.2, 0, annotated)
        
        # Draw label text
        cv2.putText(annotated, label, (label_x, label_y - score_h - 5), 
                   font, font_scale, (255, 255, 255), thickness)
        cv2.putText(annotated, score_text, (label_x, label_y), 
                   font, font_scale, (255, 255, 255), thickness)
    
    return annotated, top_boxes


def visualize_annotated_results(original_bgr, annotated_bgr, mask, anomaly_map, boxes, recon_error, save_path=None):
    """Create comprehensive visualization"""
    
    # Convert BGR to RGB for matplotlib
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Original image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(original_rgb)
    ax1.set_title('Original Image', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Anomaly heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    im = ax2.imshow(anomaly_map, cmap='jet')
    ax2.set_title(f'Anomaly Heatmap\nReconstruction Error: {recon_error:.4f}', 
                  fontsize=14, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im, ax=ax2, fraction=0.046)
    
    # Binary mask
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(mask, cmap='gray')
    ax3.set_title('Binary Segmentation Mask', fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    # Annotated image (main result)
    ax4 = fig.add_subplot(gs[1, :])
    ax4.imshow(annotated_rgb)
    ax4.set_title(f'Detected Anomalies with Bounding Boxes (Total: {len(boxes)})', 
                  fontsize=16, fontweight='bold')
    ax4.axis('off')
    
    # Add anomaly details as text
    if boxes:
        details_text = "Detected Anomalies:\n"
        for box in boxes:
            details_text += f"  #{box['id']}: Area={box['area']}px², Score={box['score']:.1f}%\n"
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax4.text(1.02, 0.5, details_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='center', bbox=props, family='monospace')
    
    plt.suptitle('Thermal Transformer Anomaly Detection Results', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization to: {save_path}")
    
    plt.show()


def detect_and_annotate(
    image_path,
    model_path='models/best_model.pth',
    threshold=0.5,
    min_area=100,
    max_area=None,
    max_annotations=3,
    blue_threshold=None,
    save_dir=None
):
    """
    Main function to detect and annotate anomalies
    
    Args:
        image_path: Path to input image
        model_path: Path to trained model
        threshold: Threshold for anomaly detection (0-1)
        min_area: Minimum area for detected anomalies (pixels)
        max_area: Maximum area for detected anomalies (pixels, None = no limit)
        max_annotations: Maximum number of top annotations to display (default: 3)
        blue_threshold: Maximum percentage of blue pixels allowed (None = no filtering)
        save_dir: Directory to save results
    
    Returns:
        annotated_image, bounding_boxes, anomaly_details
    """
    
    print("="*70)
    print("THERMAL TRANSFORMER ANOMALY DETECTION - TOP CONFIDENT ANOMALIES")
    print("="*70)
    print(f"\nInput Image: {image_path}")
    print(f"Threshold: {threshold}")
    print(f"Min Area: {min_area} pixels²")
    if max_area:
        print(f"Max Area: {max_area} pixels²")
    print(f"Max Annotations: {max_annotations} (top by confidence)")
    if blue_threshold is not None:
        print(f"Blue Filter: Remove detections with >{blue_threshold}% blue pixels")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    print("\n[1/5] Loading model...")
    model = load_model(model_path, device)
    
    # Load and preprocess image (with 10% border crop)
    print("[2/6] Loading and preprocessing image...")
    image_tensor, original_bgr, cropped_rgb, original_size, crop_coords = preprocess_image(image_path)
    image_tensor = image_tensor.to(device)
    y1, y2, x1, x2 = crop_coords
    print(f"  Original size: {original_size[1]}x{original_size[0]}")
    print(f"  Processing region: [{y1}:{y2}, {x1}:{x2}] (10% border removed)")
    
    # Generate anomaly map
    print("[3/6] Generating anomaly map...")
    with torch.no_grad():
        anomaly_map, reconstructed = model.get_anomaly_map(image_tensor)
        recon_error = torch.nn.functional.mse_loss(reconstructed, image_tensor).item()
    
    anomaly_map_np = anomaly_map.cpu().squeeze().numpy()
    print(f"  Reconstruction error: {recon_error:.6f}")
    
    # Generate binary mask
    print("[4/6] Generating segmentation mask...")
    mask = generate_anomaly_mask(anomaly_map_np, threshold=threshold)
    
    # Find contours and draw bounding boxes (mapped to full image)
    print("[5/6] Detecting and annotating anomalies...")
    print("  (Mapping detections from cropped region to full image)")
    print(f"  (Showing top {max_annotations} most confident detections)")
    if blue_threshold is not None:
        print(f"  (Filtering out detections with >{blue_threshold}% blue pixels)")
    annotated_image, boxes = find_contours_and_draw_boxes(
        original_bgr, mask, original_size, crop_coords, 
        min_area=min_area, max_area=max_area, max_annotations=max_annotations,
        blue_threshold=blue_threshold
    )
    
    print(f"\n{'='*70}")
    print(f"RESULTS: Showing top {len(boxes)} anomalies (ranked by confidence)")
    print(f"{'='*70}")
    
    if boxes:
        print("\nDetailed Anomaly Information:")
        print("-" * 70)
        for box in boxes:
            x, y, w, h = box['bbox']
            print(f"  Anomaly #{box['id']}:")
            print(f"    Location: (x={x}, y={y})")
            print(f"    Size: {w}x{h} pixels")
            print(f"    Area: {box['area']} pixels²")
            print(f"    Anomaly Score: {box['score']:.2f}%")
            if box.get('blue_pct') is not None:
                print(f"    Blue Content: {box['blue_pct']:.2f}%")
            print()
    else:
        print("\n  No anomalies detected above threshold.")
    
    # Save results
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        filename = Path(image_path).stem
        
        # Save annotated image (full original size with boxes)
        annotated_path = save_dir / f'{filename}_annotated.jpg'
        cv2.imwrite(str(annotated_path), annotated_image)
        print(f"\n[6/6] Saving results...")
        print(f"✓ Saved annotated image to: {annotated_path}")
        
        # Create full-size mask (map cropped mask to full image)
        y1, y2, x1, x2 = crop_coords
        cropped_h = y2 - y1
        cropped_w = x2 - x1
        
        # Create full-size black mask
        full_mask = np.zeros((original_size[0], original_size[1]), dtype=np.uint8)
        
        # Resize detection mask to cropped region size
        mask_cropped = cv2.resize(mask, (cropped_w, cropped_h), interpolation=cv2.INTER_NEAREST)
        
        # Place it in the corresponding region of full mask
        full_mask[y1:y2, x1:x2] = mask_cropped
        
        mask_path = save_dir / f'{filename}_mask.png'
        cv2.imwrite(str(mask_path), full_mask)
        print(f"✓ Saved mask to: {mask_path}")
        
        # Save visualization
        viz_path = save_dir / f'{filename}_visualization.png'
        visualize_annotated_results(
            original_bgr, annotated_image, mask, anomaly_map_np, 
            boxes, recon_error, save_path=viz_path
        )
        
        # Save detection report
        report_path = save_dir / f'{filename}_report.txt'
        with open(report_path, 'w') as f:
            f.write(f"Anomaly Detection Report\n")
            f.write(f"{'='*70}\n\n")
            f.write(f"Image: {image_path}\n")
            f.write(f"Image Size: {original_size[1]}x{original_size[0]}\n")
            f.write(f"Processing: 10% border removed before detection\n")
            f.write(f"Processed Region: [{y1}:{y2}, {x1}:{x2}]\n")
            f.write(f"Threshold: {threshold}\n")
            f.write(f"Reconstruction Error: {recon_error:.6f}\n")
            f.write(f"Total Anomalies Detected: {len(boxes)}\n\n")
            
            if boxes:
                f.write(f"Detected Anomalies:\n")
                f.write(f"{'-'*70}\n")
                for box in boxes:
                    x, y, w, h = box['bbox']
                    f.write(f"\nAnomaly #{box['id']}:\n")
                    f.write(f"  Bounding Box: x={x}, y={y}, width={w}, height={h}\n")
                    f.write(f"  Area: {box['area']} pixels²\n")
                    f.write(f"  Anomaly Score: {box['score']:.2f}%\n")
                    if box.get('blue_pct') is not None:
                        f.write(f"  Blue Content: {box['blue_pct']:.2f}%\n")
        
        print(f"✓ Saved report to: {report_path}")
    else:
        # Just display visualization
        visualize_annotated_results(
            original_bgr, annotated_image, mask, anomaly_map_np, 
            boxes, recon_error, save_path=None
        )
    
    return annotated_image, boxes, {
        'reconstruction_error': recon_error,
        'threshold': threshold,
        'num_anomalies': len(boxes),
        'image_size': original_size
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Detect and annotate anomalies with bounding boxes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python detect_and_annotate.py Dataset/T1/faulty/T1_faulty_001.jpg
  python detect_and_annotate.py Dataset/T1/faulty/T1_faulty_001.jpg --threshold 0.4
  python detect_and_annotate.py Dataset/T1/faulty/T1_faulty_001.jpg --threshold 0.5 --min-area 200
  python detect_and_annotate.py Dataset/T1/faulty/T1_faulty_001.jpg --save-dir custom_results
        """
    )
    
    parser.add_argument('image_path', type=str, help='Path to input thermal image')
    parser.add_argument('--model', type=str, default='models/best_model.pth', 
                       help='Path to trained model (default: models/best_model.pth)')
    parser.add_argument('--threshold', type=float, default=0.5, 
                       help='Threshold for anomaly detection, 0-1 (default: 0.5)')
    parser.add_argument('--min-area', type=int, default=100, 
                       help='Minimum area for detected anomalies in pixels (default: 100)')
    parser.add_argument('--max-area', type=int, default=None, 
                       help='Maximum area for detected anomalies in pixels (default: None, no limit)')
    parser.add_argument('--max-annotations', type=int, default=3, 
                       help='Maximum number of top confident annotations to show (default: 3)')
    parser.add_argument('--blue-threshold', type=float, default=None, 
                       help='Maximum percentage of blue pixels allowed in detection box (default: None, no filtering)')
    parser.add_argument('--save-dir', type=str, default='annotated_results', 
                       help='Directory to save results (default: annotated_results)')
    
    args = parser.parse_args()
    
    # Run detection and annotation
    annotated_img, boxes, details = detect_and_annotate(
        image_path=args.image_path,
        model_path=args.model,
        threshold=args.threshold,
        min_area=args.min_area,
        max_area=args.max_area,
        max_annotations=args.max_annotations,
        blue_threshold=args.blue_threshold,
        save_dir=args.save_dir
    )
    
    print(f"\n{'='*70}")
    print("COMPLETED!")
    print(f"{'='*70}\n")
