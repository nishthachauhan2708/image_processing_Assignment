import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def add_gaussian_noise(image, mean=0, var=0.01):
    row, col = image.shape
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col))
    noisy = image / 255 + gauss
    noisy = np.clip(noisy, 0, 1)
    noisy = np.uint8(noisy * 255)
    return noisy

def add_salt_and_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    noisy = np.copy(image)
    total_pixels = image.size
    # Salt
    num_salt = int(total_pixels * salt_prob)
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    noisy[coords[0], coords[1]] = 255
    # Pepper
    num_pepper = int(total_pixels * pepper_prob)
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    noisy[coords[0], coords[1]] = 0
    return noisy

def mean_filter(image, ksize=3):
    return cv2.blur(image, (ksize, ksize))

def median_filter(image, ksize=3):
    return cv2.medianBlur(image, ksize)

def gaussian_filter(image, ksize=3, sigma=0):
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)

def mse(orig, restored):
    return np.mean((orig.astype("float") - restored.astype("float")) ** 2)

def psnr(orig, restored):
    mse_val = mse(orig, restored)
    if mse_val == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse_val))


os.makedirs("outputs", exist_ok=True)

image_path = "image.png"  
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image not found: {image_path}")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite("outputs/original.png", gray)


gaussian_noisy = add_gaussian_noise(gray)
cv2.imwrite("outputs/gaussian_noisy.png", gaussian_noisy)

sp_noisy = add_salt_and_pepper_noise(gray)
cv2.imwrite("outputs/salt_pepper_noisy.png", sp_noisy)


# Gaussian Noise
mean_gauss = mean_filter(gaussian_noisy)
median_gauss = median_filter(gaussian_noisy)
gauss_gauss = gaussian_filter(gaussian_noisy)

# Salt & Pepper Noise
mean_sp = mean_filter(sp_noisy)
median_sp = median_filter(sp_noisy)
gauss_sp = gaussian_filter(sp_noisy)

# All filtered images
filtered_images = {
    "mean_gauss": mean_gauss,
    "median_gauss": median_gauss,
    "gauss_gauss": gauss_gauss,
    "mean_sp": mean_sp,
    "median_sp": median_sp,
    "gauss_sp": gauss_sp
}

for name, img in filtered_images.items():
    cv2.imwrite(f"outputs/{name}.png", img)


print("\nPerformance Metrics (MSE / PSNR in dB):\n")

filters_gauss = {"Mean": mean_gauss, "Median": median_gauss, "Gaussian": gauss_gauss}
filters_sp = {"Mean": mean_sp, "Median": median_sp, "Gaussian": gauss_sp}

print("Gaussian Noise:")
for name, img in filters_gauss.items():
    print(f"{name}: MSE={mse(gray,img):.2f}, PSNR={psnr(gray,img):.2f}")

print("\nSalt & Pepper Noise:")
for name, img in filters_sp.items():
    print(f"{name}: MSE={mse(gray,img):.2f}, PSNR={psnr(gray,img):.2f}")


titles = [
    "Original", "Gaussian Noise", "Salt & Pepper Noise",
    "Mean Filter (Gauss)", "Median Filter (Gauss)", "Gaussian Filter (Gauss)",
    "Mean Filter (S&P)", "Median Filter (S&P)", "Gaussian Filter (S&P)"
]

images_to_plot = [
    gray, gaussian_noisy, sp_noisy,
    mean_gauss, median_gauss, gauss_gauss,
    mean_sp, median_sp, gauss_sp
]

plt.figure(figsize=(15, 10))
for i, img in enumerate(images_to_plot):
    plt.subplot(3, 3, i+1)
    plt.imshow(img, cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.savefig("outputs/comparison_figure.png")
plt.show()

print("\nAll output images and comparison figure saved in 'outputs/' folder.")
