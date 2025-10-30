import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
import random
import matplotlib.pyplot as plt

def objective_function(I_original, thresholds):
    I_enhanced = multi_level_he(I_original, thresholds)

    try:
        psnr_value = calculate_psnr(I_original, I_enhanced, data_range=255)
    except Exception:
        return 1000.0 

    return -psnr_value

def multi_level_he(image, thresholds):
    T = sorted(list(thresholds))
    
    segments = [0] + T + [255]
    enhanced_image = np.zeros_like(image, dtype=np.uint8)
    
    for i in range(len(segments) - 1):
        t_start = segments[i]
        t_end = segments[i+1]
        
        if i == 0:
            mask = (image >= t_start) & (image <= t_end)
        else:
            mask = (image > t_start) & (image <= t_end)
        
        segment_pixels = image[mask]
        
        if len(segment_pixels) > 0:
            min_val = segment_pixels.min()
            max_val = segment_pixels.max()
            
            if min_val == max_val:
                enhanced_segment = segment_pixels 
            else:
                new_min = t_start 
                new_max = t_end  
                
                enhanced_segment = ((segment_pixels.astype(np.float32) - min_val) * (new_max - new_min) / (max_val - min_val)) + new_min
                enhanced_segment = np.clip(enhanced_segment, new_min, new_max).astype(np.uint8)
                
            enhanced_image[mask] = enhanced_segment
            
    return enhanced_image

def grey_wolf_optimizer_mlhe(image, n_thresholds=3, num_wolves=15, max_iter=50):
    print(f"Starting GWO for MLHE (Wolves: {num_wolves}, Iterations: {max_iter})...")
    
    dim = n_thresholds 
    lb = np.ones(dim) * 1 
    ub = np.ones(dim) * 254 
    
    positions = np.random.uniform(lb, ub, (num_wolves, dim))
    
    Alpha_pos = np.zeros(dim)
    Alpha_score = float('inf')
    Beta_pos = np.zeros(dim)
    Beta_score = float('inf')
    Delta_pos = np.zeros(dim)
    Delta_score = float('inf')

    psnr_history = []
    
    for l in range(max_iter):
        
        for i in range(num_wolves):
            current_thresholds = np.sort(positions[i, :])
            fitness = objective_function(image, current_thresholds) 
            
            if fitness < Alpha_score:
                Delta_score, Delta_pos = Beta_score, Beta_pos.copy()
                Beta_score, Beta_pos = Alpha_score, Alpha_pos.copy()
                Alpha_score, Alpha_pos = fitness, current_thresholds.copy()
            elif fitness < Beta_score:
                Delta_score, Delta_pos = Beta_score, Beta_pos.copy()
                Beta_score, Beta_pos = fitness, current_thresholds.copy()
            elif fitness < Delta_score:
                Delta_score, Delta_pos = fitness, current_thresholds.copy()
        
        psnr_history.append(-Alpha_score)

        a = 2 - l * (2 / max_iter)
        
        for i in range(num_wolves):
            for j in range(dim):
                r1, r2 = random.random(), random.random()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * Alpha_pos[j] - positions[i, j])

                r1, r2 = random.random(), random.random()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * Beta_pos[j] - positions[i, j])
                
                r1, r2 = random.random(), random.random()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * Delta_pos[j] - positions[i, j])

                X1 = Alpha_pos[j] - A1 * D_alpha
                X2 = Beta_pos[j] - A2 * D_beta
                X3 = Delta_pos[j] - A3 * D_delta

                positions[i, j] = (X1 + X2 + X3) / 3
                
                positions[i, j] = np.clip(positions[i, j], lb[j], ub[j])
        
        print(f"Iteration {l+1}/{max_iter}: Best PSNR = {-Alpha_score:.4f} dB, Alpha Position (T): {Alpha_pos.astype(int)}")

    print("\nGWO Complete.")
    print(f"Optimal Thresholds (T) found: {Alpha_pos.astype(int)}")
    print(f"Maximum PSNR achieved: {-Alpha_score:.4f} dB")
    
    best_enhanced_image = multi_level_he(image, Alpha_pos)
    return best_enhanced_image, image, psnr_history



try:
    img_path = 'low_contrast_image.jpg'
    original_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if original_img is None:
        print("Image not found. Creating a synthetic low-contrast image.")
        original_img = (np.random.rand(150, 150) * 80 + 90).astype(np.uint8)
except:
    print("Error loading image. Creating a synthetic low-contrast image.")
    original_img = (np.random.rand(150, 150) * 80 + 90).astype(np.uint8)


enhanced_img, original_img, history = grey_wolf_optimizer_mlhe(
    original_img, 
    n_thresholds=3, 
    num_wolves=15, 
    max_iter=50
)


fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(original_img, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(enhanced_img, cmap='gray')
axes[1].set_title(f'GWO Enhanced (Final PSNR: {history[-1]:.2f} dB)')
axes[1].axis('off')

axes[2].plot(history)
axes[2].set_title('PSNR Convergence Over Iterations')
axes[2].set_xlabel('Iteration')
axes[2].set_ylabel('PSNR (dB)')
axes[2].grid(True)

plt.tight_layout()
plt.show()