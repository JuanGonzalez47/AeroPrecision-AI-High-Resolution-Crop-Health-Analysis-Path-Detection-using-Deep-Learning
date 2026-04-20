# Inference script for high-resolution crop health and path detection
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
import cv2
from PIL import Image
import os
import matplotlib.pyplot as plt

# Path config
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'INCEPTION-V3', 'best_inception_model_script.pth')
IMAGE_DIR = os.path.join(BASE_DIR, 'data', 'raw_data','rgb-images')  

# Hiperparameters and classes
IMG_SIZE = 299
BATCH_SIZE = 64
STRIDE = 100
CLASS_NAMES = ['Crop Healthy', 'Crop Stressed', 'Path', 'Soil']

# transformation for inference (no augmentation, only resizing and normalization)
def get_inference_transform():
	return transforms.Compose([
		transforms.Resize((IMG_SIZE, IMG_SIZE)),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])

# Model
class EvolvableDroneNet(nn.Module):
	def __init__(self, num_classes):
		super().__init__()
		self.base_model = models.inception_v3(weights=None)
		self.base_model.AuxLogits.fc = nn.Linear(self.base_model.AuxLogits.fc.in_features, num_classes)
		self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)
	def forward(self, x):
		return self.base_model(x)

def load_model(device):
	model = EvolvableDroneNet(num_classes=len(CLASS_NAMES)).to(device)
	model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
	model.eval()
	return model

# Inference function that processes the image in patches and creates a path priority map
def infer_path_priority_map(image_path, model, transform, device, batch_size=BATCH_SIZE, stride=STRIDE):
	patch_size = IMG_SIZE
	img_orig = cv2.imread(image_path)
	img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
	h, w, _ = img_orig.shape
	vote_map = np.zeros((h, w, len(CLASS_NAMES)), dtype=np.float32)
	weight_accum = np.zeros((h, w), dtype=np.float32)
	kernel = np.outer(
		cv2.getGaussianKernel(patch_size, patch_size / 3),
		cv2.getGaussianKernel(patch_size, patch_size / 3)
	)
	kernel = kernel / kernel.max()
	coords = [(x, y) for y in range(0, h - patch_size + 1, stride)
					 for x in range(0, w - patch_size + 1, stride)]
	print(f"Procesando {len(coords)} zonas...")
	with torch.no_grad():
		for i in range(0, len(coords), batch_size):
			batch_coords = coords[i : i + batch_size]
			batch_tensors = [transform(Image.fromarray(img_orig[y:y+patch_size, x:x+patch_size]))
							 for x, y in batch_coords]
			inputs = torch.stack(batch_tensors).to(device)
			outputs = model(inputs)
			logits = outputs[0] if isinstance(outputs, tuple) else outputs
			probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
			for j, (x, y) in enumerate(batch_coords):
				p = probs[j].copy()
				path_idx = 2
				if p[path_idx] > 0.98:
					p[path_idx] *= 1.5
				else:
					p[path_idx] *= 0.3
				for c in range(len(CLASS_NAMES)):
					vote_map[y:y+patch_size, x:x+patch_size, c] += p[c] * kernel
				weight_accum[y:y+patch_size, x:x+patch_size] += kernel
	weight_accum[weight_accum == 0] = 1
	for c in range(len(CLASS_NAMES)):
		vote_map[:, :, c] /= weight_accum
	class_colors = {
		0: [0, 255, 0],     # Healthy - Green
		1: [255, 255, 0],   # Stressed - Yellow
		2: [101, 67, 33],   # Path - Dark Brown
		3: [255, 0, 0]      # Soil - Red
	}
	segmentation_mask = np.argmax(vote_map, axis=2)
	colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
	for cls_idx, color in class_colors.items():
		colored_mask[segmentation_mask == cls_idx] = color
	smooth_mask = cv2.GaussianBlur(colored_mask, (31, 31), 0)
	final_output = cv2.addWeighted(img_orig, 0.65, smooth_mask, 0.35, 0)
	plt.figure(figsize=(20, 12))
	plt.imshow(img_orig)
	plt.title("Original High-Resolution Image", fontsize=16)
	plt.axis('off')
	plt.figure(figsize=(20, 12))
	plt.legend(handles=[
		plt.Line2D([0], [0], marker='o', color='w', label='Healthy', markerfacecolor='green', markersize=15),
		plt.Line2D([0], [0], marker='o', color='w', label='Stressed', markerfacecolor='yellow', markersize=15),
		plt.Line2D([0], [0], marker='o', color='w', label='Path', markerfacecolor='saddlebrown', markersize=15),
		plt.Line2D([0], [0], marker='o', color='w', label='Soil', markerfacecolor='red', markersize=15)
	], loc='upper right')
	plt.imshow(final_output)
	plt.title("Cloud-Based High Precision Field Mapping", fontsize=16)
	plt.axis('off')
	plt.show()

if __name__ == '__main__':
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = load_model(device)
	transform = get_inference_transform()
	if not os.path.exists(IMAGE_DIR):
		print(f"Directorio de imágenes no encontrado: {IMAGE_DIR}")
	else:
		files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
		print(f"{len(files)} imágenes encontradas en {IMAGE_DIR}")
		for fname in files:
			print(f"Inferencia sobre: {fname}")
			image_path = os.path.join(IMAGE_DIR, fname)
			infer_path_priority_map(image_path, model, transform, device)
