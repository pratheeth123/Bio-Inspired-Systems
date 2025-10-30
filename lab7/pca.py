import numpy as np
import cv2
import matplotlib.pyplot as plt

def add_salt_and_pepper_noise(image, prob=0.05):
    """Adds salt-and-pepper noise to a grayscale image."""
    output = np.copy(image)
    total_pixels = image.size
    num_salt = np.ceil(prob * total_pixels / 2).astype(int)
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    output[coords[0], coords[1]] = 255
    num_pepper = np.ceil(prob * total_pixels / 2).astype(int)
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    output[coords[0], coords[1]] = 0
    return output

def calculate_psnr(img1, img2):
    """Calculates Peak Signal-to-Noise Ratio (PSNR) in dB."""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return 100.0 
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def parallel_ca_denoise(noisy_image, original_img, max_iterations=10, convergence_threshold=0):
    """
    Applies 2D Cellular Automata to remove Salt-and-Pepper noise, 
    tracking PSNR at each iteration.
    """
    current_image = noisy_image.copy()
    rows, cols = current_image.shape
    psnr_history = []
    for t in range(max_iterations):
        padded_image = np.pad(current_image, 1, mode='reflect')
        next_image = current_image.copy()
        changes_count = 0
        
        for r in range(rows):
            for c in range(cols):
                
                neighborhood = padded_image[r:r+3, c:c+3]
                center_pixel = neighborhood[1, 1]
                
               
                if center_pixel == 0 or center_pixel == 255:
                    
                    neighbors_without_center = np.delete(neighborhood.flatten(), 4)
                    
                  
                    non_noise_neighbors = neighbors_without_center[
                        (neighbors_without_center > 0) & (neighbors_without_center < 255)
                    ]
                    
                    new_state = center_pixel 
                    
                    if non_noise_neighbors.size > 0:
                        
                        new_state = np.median(non_noise_neighbors).astype(np.uint8)
                    else:
                        
                        new_state = np.median(neighbors_without_center).astype(np.uint8)
                        
                    
                    if new_state != center_pixel:
                        next_image[r, c] = new_state
                        changes_count += 1
                        
        
        current_image = next_image.copy()
        
       
        current_psnr = calculate_psnr(original_img, current_image)
        psnr_history.append(current_psnr)
        
       
        print(f"Iteration {t+1}: PSNR = {current_psnr:.4f} dB ({changes_count} pixels updated)")

       
        if changes_count <= convergence_threshold:
            print(f"CA converged at iteration {t+1}.")
            break
            
    return current_image, psnr_history


try:
    original_img = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)
    if original_img is None:
        raise FileNotFoundError
except FileNotFoundError:
    print("Could not load 'lena.png'. Creating a synthetic image.")
    size = 200
    original_img = np.zeros((size, size), dtype=np.uint8)
    for i in range(size):
        original_img[i, :] = np.linspace(0, 255, size).astype(np.uint8)



noise_probability = 0.15 
noisy_img = add_salt_and_pepper_noise(original_img, prob=noise_probability)


max_iters = 10 
denoised_img, psnr_history = parallel_ca_denoise(noisy_img, original_img, max_iterations=max_iters)


psnr_noisy = calculate_psnr(original_img, noisy_img)
psnr_denoised_final = psnr_history[-1] if psnr_history else calculate_psnr(original_img, denoised_img)

print("\n--- Summary ---")
print(f"Original PSNR (Noise): {psnr_noisy:.2f} dB")
print(f"Final Denoised PSNR (CA): {psnr_denoised_final:.2f} dB")


fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(original_img, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(noisy_img, cmap='gray')
axes[1].set_title(f'Noisy Image ({noise_probability*100:.0f}% S&P, PSNR: {psnr_noisy:.2f} dB)')
axes[1].axis('off')

axes[2].imshow(denoised_img, cmap='gray')
axes[2].set_title(f'CA Denoised Image (PSNR: {psnr_denoised_final:.2f} dB)')
axes[2].axis('off')

plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 4))
plt.plot(range(1, len(psnr_history) + 1), psnr_history, marker='.', linestyle='-')
plt.title('PSNR Convergence for Parallel CA Denoising')
plt.xlabel('Iteration Number')
plt.ylabel('PSNR (dB)')
plt.grid(True)
plt.show()