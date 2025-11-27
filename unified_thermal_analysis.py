#!/usr/bin/env python3
"""
Unified Thermal Transformer Analysis

Combines two analysis methods to produce a single annotated image:
1. ML-based Anomaly Detection (AutoEncoder)
2. Thermal Hotpoint Detection (Region-based)


"""

import sys
import os
import argparse
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

# Add ML_analysis to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ML_analysis'))

# Import ML detection
try:
    from ML_analysis.detect_and_annotate import (
        load_model, preprocess_image, generate_anomaly_mask,
        find_contours_and_draw_boxes
    )
    ML_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ML analysis not available: {e}")
    ML_AVAILABLE = False

# Import thermal detection
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'heat_point_analysis'))

try:
    from heat_point_analysis.thermal_hotpoint_detector import ThermalHotpointDetector
    THERMAL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Thermal analysis not available: {e}")
    THERMAL_AVAILABLE = False

import torch


class UnifiedThermalAnalyzer:
    """
    Combines ML-based and thermal-based analysis into a single output.
    """
    
    def __init__(
        self,
        ml_model_path='ML_analysis/models/best_model.pth',
        ml_threshold=0.5,
        ml_min_area=200,
        ml_max_area=5000,
        ml_max_annotations=3,
        ml_blue_threshold=30,
        thermal_temp_threshold=200,
        thermal_min_cluster_size=15,
        thermal_cluster_epsilon=20
    ):
        """
        Initialize unified analyzer.
        
        Args:
            ml_model_path: Path to ML model
            ml_threshold: ML detection threshold (0-1)
            ml_min_area: Minimum area for ML detections
            ml_max_area: Maximum area for ML detections
            ml_max_annotations: Max ML annotations to show
            ml_blue_threshold: Max blue percentage for ML
            thermal_temp_threshold: Temperature threshold for thermal detection
            thermal_min_cluster_size: Min cluster size for thermal
            thermal_cluster_epsilon: Cluster epsilon for thermal
        """
        self.ml_model_path = ml_model_path
        self.ml_threshold = ml_threshold
        self.ml_min_area = ml_min_area
        self.ml_max_area = ml_max_area
        self.ml_max_annotations = ml_max_annotations
        self.ml_blue_threshold = ml_blue_threshold
        
        self.thermal_temp_threshold = thermal_temp_threshold
        self.thermal_min_cluster_size = thermal_min_cluster_size
        self.thermal_cluster_epsilon = thermal_cluster_epsilon
        
        # Initialize components
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ml_model = None
        self.thermal_detector = None
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load both ML and thermal models."""
        print("="*70)
        print("UNIFIED THERMAL TRANSFORMER ANALYSIS")
        print("="*70)
        print(f"\nDevice: {self.device}")
        
        # Load ML model
        if ML_AVAILABLE and os.path.exists(self.ml_model_path):
            print(f"\n[1/2] Loading ML model from: {self.ml_model_path}")
            try:
                self.ml_model = load_model(self.ml_model_path, self.device)
                print("  ✓ ML model loaded successfully")
            except Exception as e:
                print(f"  ✗ Failed to load ML model: {e}")
                self.ml_model = None
        else:
            print(f"\n[1/2] ML model not available")
            self.ml_model = None
        
        # Initialize thermal detector
        if THERMAL_AVAILABLE:
            print(f"\n[2/2] Initializing thermal detector...")
            try:
                self.thermal_detector = ThermalHotpointDetector(
                    temperature_threshold=self.thermal_temp_threshold,
                    min_cluster_size=self.thermal_min_cluster_size,
                    cluster_epsilon=self.thermal_cluster_epsilon
                )
                print("  ✓ Thermal detector initialized")
            except Exception as e:
                print(f"  ✗ Failed to initialize thermal detector: {e}")
                self.thermal_detector = None
        else:
            print(f"\n[2/2] Thermal detector not available")
            self.thermal_detector = None
    
    def run_ml_analysis(self, image_path):
        """
        Run ML-based anomaly detection.
        
        Returns:
            dict with annotated_image, boxes, anomaly_map, recon_error
        """
        if self.ml_model is None:
            print("\n[ML Analysis] Skipped - model not available")
            return None
        
        print("\n" + "="*70)
        print("ML-BASED ANOMALY DETECTION")
        print("="*70)
        
        try:
            # Preprocess
            print("[1/3] Preprocessing image...")
            image_tensor, original_bgr, cropped_rgb, original_size, crop_coords = preprocess_image(image_path)
            image_tensor = image_tensor.to(self.device)
            
            # Generate anomaly map
            print("[2/3] Generating anomaly map...")
            with torch.no_grad():
                anomaly_map, reconstructed = self.ml_model.get_anomaly_map(image_tensor)
                recon_error = torch.nn.functional.mse_loss(reconstructed, image_tensor).item()
            
            anomaly_map_np = anomaly_map.cpu().squeeze().numpy()
            
            # Generate mask and find contours
            print("[3/3] Detecting anomalies...")
            mask = generate_anomaly_mask(anomaly_map_np, threshold=self.ml_threshold)
            
            annotated_image, boxes = find_contours_and_draw_boxes(
                original_bgr, mask, original_size, crop_coords,
                min_area=self.ml_min_area,
                max_area=self.ml_max_area,
                max_annotations=self.ml_max_annotations,
                blue_threshold=self.ml_blue_threshold
            )
            
            print(f"  ✓ Detected {len(boxes)} ML anomalies")
            
            return {
                'annotated_image': annotated_image,
                'boxes': boxes,
                'anomaly_map': anomaly_map_np,
                'mask': mask,
                'recon_error': recon_error,
                'original_size': original_size,
                'crop_coords': crop_coords
            }
        
        except Exception as e:
            print(f"  ✗ ML analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_thermal_analysis(self, image_path):
        """
        Run thermal hotpoint detection.
        
        Returns:
            dict with annotated_image, bounding_boxes, hot_mask
        """
        if self.thermal_detector is None:
            print("\n[Thermal Analysis] Skipped - detector not available")
            return None
        
        print("\n" + "="*70)
        print("THERMAL HOTPOINT DETECTION")
        print("="*70)
        
        try:
            print("[1/3] Loading thermal image...")
            original_image, red_channel_masked, valid_mask, crop_info, cropped_image = \
                self.thermal_detector.load_thermal_image(image_path)
            
            print("[2/3] Detecting hot regions...")
            hot_mask = self.thermal_detector.detect_hot_regions(red_channel_masked)
            
            print("[3/3] Clustering hot points...")
            clusters = self.thermal_detector.cluster_hot_points(hot_mask)
            bounding_boxes = self.thermal_detector.create_bounding_boxes(clusters, red_channel_masked.shape)
            
            print(f"  ✓ Detected {len(bounding_boxes)} thermal hotspots")
            
            # Create annotated version
            fig = self.thermal_detector.annotate_image(original_image, bounding_boxes, crop_info, show_confidence=True)
            
            # Convert figure to image
            fig.canvas.draw()
            # Use buffer_rgba() instead of tostring_rgb() for newer matplotlib
            buf = fig.canvas.buffer_rgba()
            annotated_array = np.asarray(buf)
            plt.close(fig)
            
            # Convert RGBA to BGR for OpenCV
            annotated_bgr = cv2.cvtColor(annotated_array, cv2.COLOR_RGBA2BGR)
            
            return {
                'annotated_image': annotated_bgr,
                'bounding_boxes': bounding_boxes,
                'hot_mask': hot_mask,
                'original_image': original_image,
                'crop_info': crop_info
            }
        
        except Exception as e:
            print(f"  ✗ Thermal analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_unified_visualization(self, image_path, ml_results, thermal_results, save_path=None):
        """
        Create a unified visualization combining both analyses.
        
        Args:
            image_path: Path to original image
            ml_results: Results from ML analysis
            thermal_results: Results from thermal analysis
            save_path: Path to save output
            
        Returns:
            Unified annotated image
        """
        print("\n" + "="*70)
        print("CREATING UNIFIED VISUALIZATION")
        print("="*70)
        
        # Load original image
        original_bgr = cv2.imread(image_path)
        original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
        
        # Create figure with GridSpec
        fig = plt.figure(figsize=(24, 14))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Row 1: Original + Analysis Overviews
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(original_rgb)
        ax1.set_title('Original Image', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # ML anomaly map
        if ml_results:
            ax2 = fig.add_subplot(gs[0, 1])
            im = ax2.imshow(ml_results['anomaly_map'], cmap='jet')
            ax2.set_title(f'ML Anomaly Heatmap\nRecon Error: {ml_results["recon_error"]:.4f}', 
                         fontsize=12, fontweight='bold')
            ax2.axis('off')
            plt.colorbar(im, ax=ax2, fraction=0.046)
        else:
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.text(0.5, 0.5, 'ML Analysis\nNot Available', 
                    ha='center', va='center', fontsize=14)
            ax2.axis('off')
        
        # Thermal hot mask
        if thermal_results:
            ax3 = fig.add_subplot(gs[0, 2])
            ax3.imshow(thermal_results['hot_mask'], cmap='hot')
            ax3.set_title(f'Thermal Hotpoint Map\n{len(thermal_results["bounding_boxes"])} regions', 
                         fontsize=12, fontweight='bold')
            ax3.axis('off')
        else:
            ax3 = fig.add_subplot(gs[0, 2])
            ax3.text(0.5, 0.5, 'Thermal Analysis\nNot Available', 
                    ha='center', va='center', fontsize=14)
            ax3.axis('off')
        
        # Row 2: Individual Annotations
        if ml_results:
            ax4 = fig.add_subplot(gs[1, :2])
            ml_annotated_rgb = cv2.cvtColor(ml_results['annotated_image'], cv2.COLOR_BGR2RGB)
            ax4.imshow(ml_annotated_rgb)
            ax4.set_title(f'ML Anomaly Detection ({len(ml_results["boxes"])} detections)', 
                         fontsize=13, fontweight='bold')
            ax4.axis('off')
        else:
            ax4 = fig.add_subplot(gs[1, :2])
            ax4.imshow(original_rgb)
            ax4.set_title('ML Anomaly Detection (Not Available)', fontsize=13, fontweight='bold')
            ax4.axis('off')
        
        if thermal_results:
            ax5 = fig.add_subplot(gs[1, 2])
            # Resize thermal result to match
            thermal_rgb = cv2.cvtColor(thermal_results['annotated_image'], cv2.COLOR_BGR2RGB)
            ax5.imshow(thermal_rgb)
            ax5.set_title(f'Thermal Hotpoints\n({len(thermal_results["bounding_boxes"])} hotspots)', 
                         fontsize=11, fontweight='bold')
            ax5.axis('off')
        else:
            ax5 = fig.add_subplot(gs[1, 2])
            ax5.imshow(original_rgb)
            ax5.set_title('Thermal Hotpoint Detection\n(Not Available)', fontsize=11, fontweight='bold')
            ax5.axis('off')
        
        # Row 3: Combined Result
        ax6 = fig.add_subplot(gs[2, :])
        combined_image = self._create_combined_annotation(
            original_bgr, ml_results, thermal_results
        )
        combined_rgb = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)
        ax6.imshow(combined_rgb)
        
        # Title with summary
        summary = "UNIFIED ANALYSIS: "
        if ml_results and thermal_results:
            summary += f"{len(ml_results['boxes'])} ML Anomalies + {len(thermal_results['bounding_boxes'])} Thermal Hotspots"
        elif ml_results:
            summary += f"{len(ml_results['boxes'])} ML Anomalies Only"
        elif thermal_results:
            summary += f"{len(thermal_results['bounding_boxes'])} Thermal Hotspots Only"
        else:
            summary += "No Detections"
        
        ax6.set_title(summary, fontsize=14, fontweight='bold')
        ax6.axis('off')
        
        # Add legend/summary text
        legend_text = "Detection Summary:\n"
        if ml_results:
            legend_text += f"• ML: {len(ml_results['boxes'])} anomalies (RED boxes)\n"
        if thermal_results:
            legend_text += f"• Thermal: {len(thermal_results['bounding_boxes'])} hotspots (YELLOW boxes)\n"
        
        props = dict(boxstyle='round', facecolor='lightgray', alpha=0.8)
        ax6.text(1.02, 0.5, legend_text, transform=ax6.transAxes, fontsize=11,
                verticalalignment='center', bbox=props, family='monospace')
        
        fig.suptitle(f'Unified Thermal Transformer Analysis\n{Path(image_path).name}', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"  ✓ Saved unified visualization to: {save_path}")
        
        plt.close(fig)
        
        return combined_image
    
    def _create_combined_annotation(self, original_bgr, ml_results, thermal_results):
        """
        Create a single image with both ML and thermal annotations.
        
        ML boxes: RED
        Thermal boxes: YELLOW
        """
        combined = original_bgr.copy()
        
        # Draw ML boxes (RED)
        if ml_results:
            for i, box in enumerate(ml_results['boxes'], 1):
                x, y, w, h = box['bbox']
                
                # Draw red box
                cv2.rectangle(combined, (x, y), (x + w, y + h), (0, 0, 255), 3)
                
                # Label
                label = f"ML-{i}"
                score = f"{box['score']:.1f}%"
                
                # Label background
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2
                
                (label_w, label_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
                
                # Draw label
                cv2.rectangle(combined, (x, y - label_h - 10), (x + label_w + 10, y), (0, 0, 255), -1)
                cv2.putText(combined, label, (x + 5, y - 5), font, font_scale, (255, 255, 255), thickness)
        
        # Draw thermal boxes (YELLOW)
        if thermal_results:
            for i, bbox in enumerate(thermal_results['bounding_boxes'], 1):
                # Adjust for crop
                crop_info = thermal_results['crop_info']
                x = bbox.x + crop_info['left']
                y = bbox.y + crop_info['top']
                w = bbox.width
                h = bbox.height
                
                # Draw yellow box
                cv2.rectangle(combined, (x, y), (x + w, y + h), (0, 255, 255), 3)
                
                # Label
                label = f"TH-{i}"
                conf = f"{bbox.confidence:.2f}"
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2
                
                (label_w, label_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
                
                # Draw label
                cv2.rectangle(combined, (x, y - label_h - 10), (x + label_w + 10, y), (0, 255, 255), -1)
                cv2.putText(combined, label, (x + 5, y - 5), font, font_scale, (0, 0, 0), thickness)
        
        return combined
    
    def analyze(self, image_path, output_dir='unified_results'):
        """
        Run complete unified analysis.
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save results
            
        Returns:
            dict with all results
        """
        print(f"\nInput Image: {image_path}")
        print(f"Output Directory: {output_dir}")
        
        # Run both analyses
        ml_results = self.run_ml_analysis(image_path)
        thermal_results = self.run_thermal_analysis(image_path)
        
        # Create unified visualization
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = Path(image_path).stem
        save_path = output_dir / f'{filename}_unified_analysis.png'
        
        combined_image = self.create_unified_visualization(
            image_path, ml_results, thermal_results, save_path=save_path
        )
        
        # Save combined annotated image
        combined_path = output_dir / f'{filename}_combined_annotated.jpg'
        cv2.imwrite(str(combined_path), combined_image)
        print(f"  ✓ Saved combined annotation to: {combined_path}")
        
        # Save report
        report_path = output_dir / f'{filename}_unified_report.txt'
        self._save_report(report_path, image_path, ml_results, thermal_results)
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE!")
        print("="*70)
        print(f"\nResults saved to: {output_dir}/")
        print(f"  • Visualization: {save_path.name}")
        print(f"  • Combined Image: {combined_path.name}")
        print(f"  • Report: {report_path.name}")
        
        return {
            'ml_results': ml_results,
            'thermal_results': thermal_results,
            'combined_image': combined_image,
            'output_dir': output_dir
        }
    
    def _save_report(self, report_path, image_path, ml_results, thermal_results):
        """Save text report."""
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("UNIFIED THERMAL TRANSFORMER ANALYSIS REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Image: {image_path}\n\n")
            
            # ML Results
            f.write("ML-BASED ANOMALY DETECTION\n")
            f.write("-"*70 + "\n")
            if ml_results:
                f.write(f"Status: SUCCESS\n")
                f.write(f"Reconstruction Error: {ml_results['recon_error']:.6f}\n")
                f.write(f"Threshold: {self.ml_threshold}\n")
                f.write(f"Detected Anomalies: {len(ml_results['boxes'])}\n\n")
                
                if ml_results['boxes']:
                    f.write("Anomalies:\n")
                    for box in ml_results['boxes']:
                        x, y, w, h = box['bbox']
                        f.write(f"  ML-{box['id']}:\n")
                        f.write(f"    Location: ({x}, {y})\n")
                        f.write(f"    Size: {w}x{h}\n")
                        f.write(f"    Score: {box['score']:.2f}%\n")
                        if box.get('blue_pct'):
                            f.write(f"    Blue Content: {box['blue_pct']:.2f}%\n")
                        f.write("\n")
            else:
                f.write(f"Status: NOT AVAILABLE\n")
            
            f.write("\n")
            
            # Thermal Results
            f.write("THERMAL HOTPOINT DETECTION\n")
            f.write("-"*70 + "\n")
            if thermal_results:
                f.write(f"Status: SUCCESS\n")
                f.write(f"Temperature Threshold: {self.thermal_temp_threshold}\n")
                f.write(f"Detected Hotspots: {len(thermal_results['bounding_boxes'])}\n\n")
                
                if thermal_results['bounding_boxes']:
                    f.write("Hotspots:\n")
                    for i, bbox in enumerate(thermal_results['bounding_boxes'], 1):
                        f.write(f"  TH-{i}:\n")
                        f.write(f"    Location: ({bbox.x}, {bbox.y})\n")
                        f.write(f"    Size: {bbox.width}x{bbox.height}\n")
                        f.write(f"    Confidence: {bbox.confidence:.2f}\n")
                        f.write("\n")
            else:
                f.write(f"Status: NOT AVAILABLE\n")
            
            f.write("\n" + "="*70 + "\n")
        
        print(f"  ✓ Saved report to: {report_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Unified Thermal Transformer Analysis - Combines ML and Thermal Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python unified_thermal_analysis.py Dataset/T1/faulty/T1_faulty_001.jpg
  
  python unified_thermal_analysis.py Dataset/T1/faulty/T1_faulty_001.jpg \\
      --threshold 0.5 --min-area 200 --max-area 5000 \\
      --max-annotations 3 --blue-threshold 30
  
  python unified_thermal_analysis.py Dataset/T1/faulty/T1_faulty_001.jpg \\
      --output-dir my_results --thermal-threshold 180
        """
    )
    
    parser.add_argument('image_path', type=str, help='Path to thermal transformer image')
    
    # ML parameters
    parser.add_argument('--model', type=str, default='ML_analysis/models/best_model.pth',
                       help='Path to ML model (default: ML_analysis/models/best_model.pth)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='ML detection threshold 0-1 (default: 0.5)')
    parser.add_argument('--min-area', type=int, default=200,
                       help='Min area for ML detections (default: 200)')
    parser.add_argument('--max-area', type=int, default=5000,
                       help='Max area for ML detections (default: 5000)')
    parser.add_argument('--max-annotations', type=int, default=3,
                       help='Max ML annotations (default: 3)')
    parser.add_argument('--blue-threshold', type=float, default=30,
                       help='Max blue percentage for ML (default: 30)')
    
    # Thermal parameters
    parser.add_argument('--thermal-threshold', type=int, default=200,
                       help='Temperature threshold for thermal detection (default: 200)')
    parser.add_argument('--thermal-min-cluster', type=int, default=15,
                       help='Min cluster size for thermal (default: 15)')
    parser.add_argument('--thermal-epsilon', type=int, default=20,
                       help='Cluster epsilon for thermal (default: 20)')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='unified_results',
                       help='Output directory (default: unified_results)')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = UnifiedThermalAnalyzer(
        ml_model_path=args.model,
        ml_threshold=args.threshold,
        ml_min_area=args.min_area,
        ml_max_area=args.max_area,
        ml_max_annotations=args.max_annotations,
        ml_blue_threshold=args.blue_threshold,
        thermal_temp_threshold=args.thermal_threshold,
        thermal_min_cluster_size=args.thermal_min_cluster,
        thermal_cluster_epsilon=args.thermal_epsilon
    )
    
    # Run analysis
    results = analyzer.analyze(args.image_path, args.output_dir)


if __name__ == '__main__':
    main()
