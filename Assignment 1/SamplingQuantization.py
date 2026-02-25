
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

print("="*70)
print("           SMART DOCUMENT SCANNER & QUALITY ANALYSIS SYSTEM")
print("           Course: Image Processing & Computer Vision")
print("           Simulates image acquisition, sampling & quantization effects")
print("="*70)
print(f"Run Date & Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70)


if not os.path.exists("outputs"):
    os.makedirs("outputs")
    print("\n[OK] Created 'outputs' directory for saving results")


def load_and_preprocess(image_path, image_id=1):
    """
    Load document image, resize to 512x512, convert to grayscale
    """
    print(f"\n{'='*50}")
    print(f"[DOCUMENT {image_id}]: {os.path.basename(image_path)}")
    print(f"{'='*50}")
    

    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Could not load image from {image_path}")
        return None, None, None
    
  
    h, w = img.shape[:2]
    print(f"[INFO] Original dimensions: {w} x {h} pixels")
    
  
    img_resized = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
    
  
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    print(f"[OK] Resized to: 512 x 512 pixels")
    print(f"[OK] Converted to grayscale (8-bit)")
    
    return img_resized, gray, image_id


def analyze_sampling(gray_image):
    """
    Downsample to different resolutions and upsample back for comparison
    """
    print("\n--- TASK 3: SAMPLING ANALYSIS (Resolution Reduction) ---")
    
    resolutions = [512, 256, 128]
    labels = ["High (512x512)", "Medium (256x256)", "Low (128x128)"]
    sampled_images = []
    
    for i, (res, label) in enumerate(zip(resolutions, labels)):
        
        downsampled = cv2.resize(gray_image, (res, res), interpolation=cv2.INTER_AREA)
        
        
        upsampled = cv2.resize(downsampled, (512, 512), interpolation=cv2.INTER_LINEAR)
        sampled_images.append(upsampled)
        
        
        cv2.imwrite(f"outputs/sampled_{res}x{res}.png", downsampled)
        
        print(f"   [OK] {label}: {res}x{res} pixels (saved)")
    
    return sampled_images


def quantize_image(gray_image, levels):
    """
    Reduce number of gray levels
    """
    step = 256 // levels
    quantized = (gray_image // step) * step
    return quantized.astype(np.uint8)

def analyze_quantization(gray_image):
    """
    Quantize to different bit depths
    """
    print("\n--- TASK 4: QUANTIZATION ANALYSIS (Bit-depth Reduction) ---")
    
    
    bit_depths = [8, 4, 2]
    gray_levels = [256, 16, 4]
    labels = ["8-bit (256 levels)", "4-bit (16 levels)", "2-bit (4 levels)"]
    quantized_images = []
    
    
    quantized_images.append(gray_image)
    cv2.imwrite("outputs/quantized_8bit.png", gray_image)
    print(f"   [OK] {labels[0]} (saved)")
    

    q_16 = quantize_image(gray_image, 16)
    quantized_images.append(q_16)
    cv2.imwrite("outputs/quantized_4bit.png", q_16)
    print(f"   [OK] {labels[1]} (saved)")
    
  
    q_4 = quantize_image(gray_image, 4)
    quantized_images.append(q_4)
    cv2.imwrite("outputs/quantized_2bit.png", q_4)
    print(f"   [OK] {labels[2]} (saved)")
    
    return quantized_images


def create_comparison_figure(original, sampled_images, quantized_images, image_id):
    """
    Create a 2x3 comparison figure showing all results
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Document Scanner Quality Analysis - Document {image_id}', fontsize=16, fontweight='bold')
    
    
    axes[0,0].imshow(original, cmap='gray')
    axes[0,0].set_title("ORIGINAL\n(512x512, 8-bit)", fontsize=12, fontweight='bold')
    axes[0,0].axis('off')
    
    axes[0,1].imshow(sampled_images[1], cmap='gray')
    axes[0,1].set_title("SAMPLED: Medium Resolution\n(256x256)", fontsize=12, fontweight='bold')
    axes[0,1].axis('off')
    
    axes[0,2].imshow(sampled_images[2], cmap='gray')
    axes[0,2].set_title("SAMPLED: Low Resolution\n(128x128)", fontsize=12, fontweight='bold')
    axes[0,2].axis('off')
    
    
    axes[1,0].imshow(quantized_images[0], cmap='gray')
    axes[1,0].set_title("QUANTIZED: 8-bit\n(256 gray levels)", fontsize=12, fontweight='bold')
    axes[1,0].axis('off')
    
    axes[1,1].imshow(quantized_images[1], cmap='gray')
    axes[1,1].set_title("QUANTIZED: 4-bit\n(16 gray levels)", fontsize=12, fontweight='bold')
    axes[1,1].axis('off')
    
    axes[1,2].imshow(quantized_images[2], cmap='gray')
    axes[1,2].set_title("QUANTIZED: 2-bit\n(4 gray levels)", fontsize=12, fontweight='bold')
    axes[1,2].axis('off')
    
    plt.tight_layout()
    
    
    comparison_path = f"outputs/comparison_doc{image_id}.png"
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Comparison figure saved: {comparison_path}")
    
    return fig

def print_observations():
    """
    Print detailed quality analysis observations
    """
    print("\n" + "="*70)
    print("TASK 5: QUALITY OBSERVATIONS & ANALYSIS")
    print("="*70)
    
    print("\nTEXT CLARITY ANALYSIS:")
    print("   * 512x512 (High Resolution):")
    print("     - Text is sharp and crisp")
    print("     - All character edges are well-defined")
    print("     - Fine details like punctuation visible")
    print("   * 256x256 (Medium Resolution):")
    print("     - Slight blurring observed")
    print("     - Main text remains readable")
    print("     - Small fonts start losing sharpness")
    print("   * 128x128 (Low Resolution):")
    print("     - Significant blurring")
    print("     - Character edges become jagged")
    print("     - Small text becomes illegible")
    
    print("\nREADABILITY DEGRADATION:")
    print("   * 8-bit (256 levels):")
    print("     - Perfect readability, original quality preserved")
    print("   * 4-bit (16 levels):")
    print("     - Visible false contours in smooth regions")
    print("     - Text remains readable but quality reduced")
    print("   * 2-bit (4 levels):")
    print("     - Heavy posterization effects")
    print("     - Significant loss of gray-scale information")
    print("     - Text barely readable, severe quality degradation")
    
    print("\nOCR SUITABILITY ASSESSMENT:")
    print("   * HIGH SUITABILITY: 512x512 & 8-bit")
    print("     - Ideal for OCR engines")
    print("     - Maximum accuracy expected")
    print("   * MODERATE SUITABILITY: 256x256 & 4-bit")
    print("     - May work with preprocessing")
    print("     - Some errors possible with small text")
    print("   * LOW SUITABILITY: 128x128 & 2-bit")
    print("     - Not recommended for OCR")
    print("     - High error rate expected")
    
    print("\nRECOMMENDATIONS:")
    print("   * For archival: Use 512x512, 8-bit minimum")
    print("   * For OCR processing: Minimum 300 DPI scan recommended")
    print("   * For web display: 256x256, 4-bit may be acceptable")
    print("="*70)


def process_document(image_path, image_id):
    """
    Process a single document through all tasks
    """
  
    orig_color, gray, doc_id = load_and_preprocess(image_path, image_id)
    
    if orig_color is None:
        return False
    
  
    cv2.imwrite(f"outputs/grayscale_doc{image_id}.png", gray)
    
  
    sampled = analyze_sampling(gray)
  
    quantized = analyze_quantization(gray)
    
    fig = create_comparison_figure(gray, sampled, quantized, image_id)
    plt.show()
    
    return True


if __name__ == "__main__":

    document_paths = [
        "document1.jpeg",      
        "document2.jpeg",      
        "document3.jpeg"       
    ]
    
    print("\nDOCUMENTS TO PROCESS:")
    for i, path in enumerate(document_paths, 1):
        if os.path.exists(path):
            status = "Found"
        else:
            status = "Not Found"
        print(f"   Document {i}: {path} [{status}]")
    
  
    successful = 0
    for i, doc_path in enumerate(document_paths, 1):
        if os.path.exists(doc_path):
            print(f"\n{'='*60}")
            print(f"Processing Document {i}...")
            print(f"{'='*60}")
            if process_document(doc_path, i):
                successful += 1
        else:
            print(f"\n[WARNING] Document {i} not found at {doc_path}")
    
    
    if successful > 0:
        print_observations()
    
    
    print(f"\n{'='*70}")
    print("PROCESSING SUMMARY:")
    print(f"   Successfully processed: {successful} document(s)")
    print(f"   Outputs saved in: outputs/")
    print(f"\nGenerated files in outputs folder:")
    print(f"   - grayscale_doc*.png")
    print(f"   - sampled_*x*.png")
    print(f"   - quantized_*bit.png")
    print(f"   - comparison_doc*.png")
    print(f"{'='*70}")
    print("\nAssignment completed!")
    print("Remember to:")
    print("   * Add your name and roll number in header")
    print("   * Push code to GitHub repository")
    print("   * Submit repository URL via LMS")
    print("="*70)
