# Entrenamiento de InceptionV3 con mejores hiperparámetros y rutas locales
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# paths config

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'dataset_model')
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'models', 'INCEPTION-V3', 'best_inception_model_script.pth')

# Optim Hiperparamters (See notebook pipeline_train_model.ipynb for details on how these were determined)
best_params = {
	'arch': 'inception',
	'lr': 0.000548,
	'weight_decay': 0.000104,
	'batch_size': 64,
	'depth_factor': 1,
	'width_multiplier': 128,
	'dropout_rate': 0.333,
	'freeze_backbone': False
}

# Transformations
def get_data_transforms(img_size):
	return {
		'train': transforms.Compose([
			transforms.Resize((img_size, img_size)),
			transforms.RandomHorizontalFlip(),
			transforms.RandomVerticalFlip(),
			transforms.RandomRotation(20),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		]),
		'val': transforms.Compose([
			transforms.Resize((img_size, img_size)),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		]),
	}

# Model EvolvableDroneNet
class EvolvableDroneNet(nn.Module):
	def __init__(self, arch_name, num_classes, depth_factor=1, width_multiplier=512, dropout_rate=0.3):
		super(EvolvableDroneNet, self).__init__()
		if arch_name == 'inception':
			self.base_model = models.inception_v3(weights='IMAGENET1K_V1')
			self.base_model.AuxLogits.fc = nn.Linear(self.base_model.AuxLogits.fc.in_features, num_classes)
			self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)
		else:
			raise ValueError('Solo se soporta InceptionV3 en este pipeline')
	def forward(self, x):
		return self.base_model(x)

def main():
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	img_size = 299
	data_transforms = get_data_transforms(img_size)

	# Datasets and Dataloaders
	print("Loading datasets...")
	train_dataset = datasets.ImageFolder(os.path.join(DATA_PATH, 'train'), transform=data_transforms['train'])
	val_dataset = datasets.ImageFolder(os.path.join(DATA_PATH, 'val'), transform=data_transforms['val'])
	train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
	val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'], shuffle=False)
	class_names = train_dataset.classes
	print(f"Datasets loaded: {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")

	# Model
	print("Initializing model...")
	model = EvolvableDroneNet(
		arch_name=best_params['arch'],
		num_classes=len(class_names),
		depth_factor=best_params['depth_factor'],
		width_multiplier=best_params['width_multiplier'],
		dropout_rate=best_params['dropout_rate']
	).to(device)
	print("Model initialized.")

	if best_params['freeze_backbone']:
		for param in model.base_model.parameters():
			param.requires_grad = False
		for param in model.base_model.fc.parameters():
			param.requires_grad = True
		for param in model.base_model.AuxLogits.fc.parameters():
			param.requires_grad = True
	
	optimizer = torch.optim.Adam(
		filter(lambda p: p.requires_grad, model.parameters()),
		lr=best_params['lr'],
		weight_decay=best_params['weight_decay']
	)
	criterion = nn.CrossEntropyLoss()

	num_epochs = 30
	patience = 7
	best_val_acc = 0.0
	epochs_without_improvement = 0

	print("Starting training...")
	# Training loop
	for epoch in range(num_epochs):
		model.train()
		running_loss, running_corrects, total = 0.0, 0, 0
		for inputs, labels in train_loader:
			inputs, labels = inputs.to(device), labels.to(device)
			optimizer.zero_grad()
			outputs = model(inputs)
			logits, aux_logits = outputs
			loss_main = criterion(logits, labels)
			loss_aux = criterion(aux_logits, labels)
			loss = loss_main + 0.4 * loss_aux
			preds = torch.argmax(logits, 1)
			loss.backward()
			optimizer.step()
			running_loss += loss.item() * inputs.size(0)
			running_corrects += torch.sum(preds == labels.data)
			total += labels.size(0)
		epoch_train_loss = running_loss / total
		epoch_train_acc = running_corrects.double() / total

		# Validation
		model.eval()
		running_loss, running_corrects, total = 0.0, 0, 0
		with torch.no_grad():
			for inputs, labels in val_loader:
				inputs, labels = inputs.to(device), labels.to(device)
				outputs = model(inputs)
				logits = outputs[0] if isinstance(outputs, tuple) else outputs
				loss = criterion(logits, labels)
				preds = torch.argmax(logits, 1)
				running_loss += loss.item() * inputs.size(0)
				running_corrects += torch.sum(preds == labels.data)
				total += labels.size(0)
		epoch_val_loss = running_loss / total
		epoch_val_acc = running_corrects.double() / total

		print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")

		# Early stopping
		if epoch_val_acc > best_val_acc:
			best_val_acc = epoch_val_acc
			epochs_without_improvement = 0
			os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
			torch.save(model.state_dict(), MODEL_SAVE_PATH)
			print(f"Modelo guardado en {MODEL_SAVE_PATH}")
		else:
			epochs_without_improvement += 1
		if epochs_without_improvement >= patience:
			print(f"Early stopping at epoch {epoch+1}")
			break

if __name__ == '__main__':
	main()
