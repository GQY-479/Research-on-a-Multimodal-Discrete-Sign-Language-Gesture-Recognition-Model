from c2net.context import prepare,upload_output

#初始化导入数据集和预训练模型到容器内
c2net_context = prepare()

#获取数据集路径
emg_data_path = c2net_context.dataset_path+"/"+"emg_data"
imu_data_path = c2net_context.dataset_path+"/"+"imu_data"

imu_data_path = c2net_context.dataset_path+"/"+"imu_data"+"/"+"imu_data"
emg_data_path = c2net_context.dataset_path+"/"+"emg_data"+"/"+"emg_data"

#输出结果必须保存在该目录
you_should_save_here = c2net_context.output_path
# save_path = you_should_save_here+"/"+"model.ckpt"
save_path = you_should_save_here

import numpy as np
import torch
import random

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# set_seed(87)
set_seed(123)


import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os

class FusedDataset(Dataset):
    def __init__(self, emg_directory, imu_directory, emg_seq_length=790, imu_seq_length=205):
        self.emg_seq_length = emg_seq_length
        self.imu_seq_length = imu_seq_length
        self.emg_files = []
        self.imu_files = []
        self.labels = []

        # Load EMG and IMU files into dictionaries with full filenames (minus extension) as keys
        emg_files = {f[:-8]: os.path.join(emg_directory, f) for f in os.listdir(emg_directory) if f.endswith("_emg.txt")}
        imu_files = {f[:-8]: os.path.join(imu_directory, f) for f in os.listdir(imu_directory) if f.endswith("_imu.txt")}

        # Match EMG and IMU files based on the same label (derived from full filenames)
        for file_key in emg_files:
            if file_key in imu_files:
                emg_filepath = emg_files[file_key]
                imu_filepath = imu_files[file_key]
                if os.path.getsize(emg_filepath) > 0 and os.path.getsize(imu_filepath) > 0:
                    self.emg_files.append(emg_filepath)
                    self.imu_files.append(imu_filepath)
                    self.labels.append(file_key.split('_')[0])

        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.labels)
        self.labels = torch.from_numpy(self.labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        emg_path = self.emg_files[idx]
        imu_path = self.imu_files[idx]
        emg_signals = self.load_and_process_data(emg_path, self.emg_seq_length)
        imu_signals = self.load_and_process_data(imu_path, self.imu_seq_length)
        
        return emg_signals, imu_signals, self.labels[idx]

    def load_and_process_data(self, filepath, seq_length):
        data = np.loadtxt(filepath)
        if data.shape[0] < seq_length:
            data = np.vstack([data, np.zeros((seq_length - data.shape[0], data.shape[1]))])
        elif data.shape[0] > seq_length:
            data = data[:seq_length, :]
        return torch.from_numpy(data).float()

    def get_num_classes(self):
        return len(np.unique(self.labels))



# from your_dataset_file import FusedDataset
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

def get_fused_dataloader(emg_dir, imu_dir, batch_size, n_workers, random_state=42, fix_val_seed=False):
    dataset = FusedDataset(emg_directory=emg_dir, imu_directory=imu_dir)
    num_classes = dataset.get_num_classes()

    # Ensure each class has at least 10% samples in the test set and at least one sample
    labels = dataset.labels.numpy()
    train_val_indices, test_indices = [], []

    rng = np.random.default_rng(random_state)
    for label in np.unique(labels):
        label_indices = np.where(labels == label)[0]
        test_size = max(1, int(0.1 * len(label_indices)))
        test_idx = rng.choice(label_indices, size=test_size, replace=False)
        train_val_idx = np.setdiff1d(label_indices, test_idx)

        test_indices.extend(test_idx)
        train_val_indices.extend(train_val_idx)

    # Split the remaining data into training and validation sets
    train_val_labels = labels[train_val_indices]
    if fix_val_seed:
        train_val_indices, valid_indices = train_test_split(
            train_val_indices, test_size=0.1, stratify=train_val_labels, random_state=random_state
        )
    else:
        train_val_indices, valid_indices = train_test_split(
            train_val_indices, test_size=0.1, stratify=train_val_labels
        )

    train_set = Subset(dataset, train_val_indices)
    valid_set = Subset(dataset, valid_indices)
    test_set = Subset(dataset, test_indices)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=n_workers,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        num_workers=n_workers,
        drop_last=True,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        num_workers=n_workers,
        drop_last=True,
        pin_memory=True,
    )

    return train_loader, valid_loader, test_loader, num_classes



import torch.nn as nn

class SignLanguageModel(nn.Module):
    def __init__(self, num_classes, emg_input_dim=8, imu_input_dim=10, hidden_dim=128, num_heads=2, num_layers=1, feature_dim=128, post_fusion_layers=1):
        super(SignLanguageModel, self).__init__()
        self.emg_embedding = nn.Linear(emg_input_dim, hidden_dim)
        self.imu_embedding = nn.Linear(imu_input_dim, hidden_dim)
        
        # Adjusting the number of heads to ensure it divides the embedding dimensions
        self.emg_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=emg_input_dim, nhead=2, dim_feedforward=emg_input_dim*4, batch_first=True),
            num_layers=num_layers
        )
        self.imu_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=imu_input_dim, nhead=2, dim_feedforward=imu_input_dim*4, batch_first=True),
            num_layers=num_layers
        )

        self.fusion_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=feature_dim * 2, nhead=num_heads, dim_feedforward=feature_dim*4, batch_first=True),
            num_layers=post_fusion_layers
        )

        self.fc_dim_reduction = nn.Linear(feature_dim * 2, feature_dim)  # Optional: Reducing dimension before final classification
        self.fc_final = nn.Linear(feature_dim, num_classes)  # Final classification layer

    def forward(self, emg_data, imu_data):
        emg_features = self.emg_transformer(emg_data)
        imu_features = self.imu_transformer(imu_data)

        emg_features = self.emg_embedding(emg_features)
        imu_features = self.imu_embedding(imu_features)        
        
        emg_pooled = torch.mean(emg_features, dim=1)
        imu_pooled = torch.mean(imu_features, dim=1)

        combined_features = torch.cat((emg_pooled, imu_pooled), dim=1)
        combined_features = self.fusion_transformer(combined_features.unsqueeze(1)).squeeze(1)
        reduced_features = self.fc_dim_reduction(combined_features)
        
        output = self.fc_final(reduced_features)
        return output

class SignLanguageModelWithLSTM(nn.Module):
    def __init__(self, num_classes, emg_input_dim=8, imu_input_dim=10, hidden_dim=128, num_heads=2, num_layers=1, feature_dim=128, post_fusion_layers=1):
        super(SignLanguageModelWithLSTM, self).__init__()
        
        self.emg_lstm = nn.LSTM(emg_input_dim, hidden_dim, batch_first=True)
        self.imu_lstm = nn.LSTM(imu_input_dim, hidden_dim, batch_first=True)

        self.fusion_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim * 2, nhead=num_heads, dim_feedforward=hidden_dim * 4, batch_first=True),
            num_layers=post_fusion_layers
        )

        self.fc_dim_reduction = nn.Linear(hidden_dim * 2, feature_dim)
        self.fc_final = nn.Linear(feature_dim, num_classes)

    def forward(self, emg_data, imu_data):
        emg_features, _ = self.emg_lstm(emg_data)
        imu_features, _ = self.imu_lstm(imu_data)

        emg_pooled = torch.mean(emg_features, dim=1)
        imu_pooled = torch.mean(imu_features, dim=1)

        combined_features = torch.cat((emg_pooled, imu_pooled), dim=1)
        combined_features = self.fusion_transformer(combined_features.unsqueeze(1)).squeeze(1)
        reduced_features = self.fc_dim_reduction(combined_features)
        
        output = self.fc_final(reduced_features)
        return output

import math

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(
	optimizer: Optimizer,
	num_warmup_steps: int,
	num_training_steps: int,
	num_cycles: float = 0.5,
	last_epoch: int = -1,
):
	"""
	Create a schedule with a learning rate that decreases following the values of the cosine function between the
	initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
	initial lr set in the optimizer.

	Args:
		optimizer (:class:`~torch.optim.Optimizer`):
		The optimizer for which to schedule the learning rate.
		num_warmup_steps (:obj:`int`):
		The number of steps for the warmup phase.
		num_training_steps (:obj:`int`):
		The total number of training steps.
		num_cycles (:obj:`float`, `optional`, defaults to 0.5):
		The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
		following a half-cosine).
		last_epoch (:obj:`int`, `optional`, defaults to -1):
		The index of the last epoch when resuming training.

	Return:
		:obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
	"""
	def lr_lambda(current_step):
		# Warmup
		if current_step < num_warmup_steps:
			return float(current_step) / float(max(1, num_warmup_steps))
		# decadence
		progress = float(current_step - num_warmup_steps) / float(
			max(1, num_training_steps - num_warmup_steps)
		)
		return max(
			0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
		)

	return LambdaLR(optimizer, lr_lambda, last_epoch)



def model_fn(batch, model, criterion, device):
    emg_data, imu_data, labels = batch
    emg_data = emg_data.to(device)
    imu_data = imu_data.to(device)
    labels = labels.to(device)

    outputs = model(emg_data, imu_data)
    loss = criterion(outputs, labels)

    preds = outputs.argmax(dim=1)
    accuracy = (preds == labels).float().mean()

    return loss, accuracy




def valid(dataloader, model, criterion, device):
    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0
    pbar = tqdm(total=len(dataloader.dataset), ncols=0, desc="Valid", unit="sample")

    for i, batch in enumerate(dataloader):
        with torch.no_grad():
            loss, accuracy = model_fn(batch, model, criterion, device)
            running_loss += loss.item()
            running_accuracy += accuracy.item()

        pbar.update(dataloader.batch_size)
        pbar.set_postfix(
            loss=f"{running_loss / (i+1):.2f}",
            accuracy=f"{running_accuracy / (i+1):.2f}",
        )

    pbar.close()
    model.train()

    return running_loss / len(dataloader), running_accuracy / len(dataloader)


import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
import numpy as np

def main(
    emg_dir,
    imu_dir,
    save_path,
    batch_size,
    n_workers,
    valid_steps,
    warmup_steps,
    total_steps,
    save_steps,
    pretrained_path=None,
    early_stop=5,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info]: Use {device} now!")

    train_loader, valid_loader, test_loader, num_classes = get_fused_dataloader(emg_dir, imu_dir, batch_size, n_workers)
    print(f"[Info]: Finish loading data!", flush=True)

    # model = SignLanguageModel(num_classes=num_classes).to(device)
    model = SignLanguageModelWithLSTM(num_classes=num_classes).to(device)

    if pretrained_path:
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
        print("[Info]: Pretrained model loaded!")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-3*4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Lists to store metrics
    train_losses = []
    train_accuracies = []
    valid_losses = []
    valid_accuracies = []

    best_accuracy = -1.0
    best_state_dict = None
    no_improve_epochs = 0  # Count epochs with no improvement

    pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")

    for step in range(total_steps):
        try:
            batch = next(iter(train_loader))
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)

        loss, accuracy = model_fn(batch, model, criterion, device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Store metrics
        train_losses.append(loss.item())
        train_accuracies.append(accuracy.item())

        pbar.update()
        pbar.set_postfix(
            loss=f"{loss.item():.2f}",
            accuracy=f"{accuracy:.2f}",
            step=step + 1,
        )

        if (step + 1) % valid_steps == 0:
            pbar.close()
            valid_loss, valid_accuracy = valid(valid_loader, model, criterion, device)
            valid_losses.append(valid_loss)
            valid_accuracies.append(valid_accuracy)

            if valid_accuracy > best_accuracy:
                best_accuracy = valid_accuracy
                best_state_dict = model.state_dict()
                no_improve_epochs = 0  # Reset counter
            else:
                no_improve_epochs += 1

            if no_improve_epochs >= early_stop:
                print("Early stopping triggered due to no improvement in validation accuracy.")
                break  # Break the loop if no improvement in the last 'early_stop' epochs                

            pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")

        if (step + 1) % save_steps == 0 and best_state_dict is not None:
            torch.save(best_state_dict, save_path+"/"+"model.ckpt")
            print(f"Step {step + 1}, best model saved. (accuracy={best_accuracy:.4f})")

    pbar.close()

    # Plot and save loss
    plt.figure(figsize=(10, 4))
#     plt.plot(np.arange(len(train_losses)) * valid_steps, train_losses, label='Train Loss')
    plt.plot(train_losses, label='Train Loss')
    plt.plot(np.arange(1, len(valid_losses) + 1) * valid_steps, valid_losses, label='Valid Loss')
    plt.title('Loss during Training')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path + "/loss_plot.png")
    plt.show()

    # Plot and save accuracy
    plt.figure(figsize=(10, 4))
#     plt.plot(np.arange(len(train_accuracies)) * valid_steps, train_accuracies, label='Train Accuracy')
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(np.arange(1, len(valid_accuracies) + 1) * valid_steps, valid_accuracies, label='Valid Accuracy')
    plt.title('Accuracy during Training')
    plt.xlabel('Steps')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(save_path + "/accuracy_plot.png")
    plt.show()

    # Save metrics to files
    np.savetxt(save_path + "/train_losses.txt", np.array(train_losses))
    np.savetxt(save_path + "/train_accuracies.txt", np.array(train_accuracies))
    np.savetxt(save_path + "/valid_losses.txt", np.array(valid_losses))
    np.savetxt(save_path + "/valid_accuracies.txt", np.array(valid_accuracies))

    #回传结果到openi，只有训练任务才能回传
    upload_output()
    


def parse_args():
    """arguments"""
    config = {
        # "data_dir": "./emg_data",
        # "data_dir": "/tmp/dataset/emg_data/emg_data",
        # "emg_dir": "/tmp/dataset/emg_data/emg_data",
        # "imu_dir": "/tmp/dataset/imu_data/imu_data",
        "emg_dir": emg_data_path,
        "imu_dir": imu_data_path,
        "save_path": save_path,
        "batch_size": 128,
        "n_workers": 8,
        "valid_steps": 500,
        "warmup_steps": 500,
        "save_steps": 500,
        "total_steps": 1500,
        "early_stop": 6,
        "pretrained_path": None,
        # "pretrained_path": "model-pretrained-0.9213.ckpt",  # 可以设置为预先训练好的模型路径
    }
    return config


if __name__ == "__main__":
    main(**parse_args())

