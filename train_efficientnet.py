import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms, models
from torch import nn, optim
from PIL import Image
from collections import defaultdict
from tqdm import tqdm  # 引入 tqdm

# 1. 設定超參數
train_txt = "../../dataset/classifier-20241129/train.txt"
val_txt = "../../dataset/classifier-20241129/val.txt"

batch_size = 32
learning_rate = 0.001
num_epochs = 50
val_split = 0.2  # 驗證集比例

# [lympho, mono, other]
num_classes = 3


# 自定義 Dataset
class CustomDataset(Dataset):

    def __init__(self, txt_file, transform=None):
        self.data = []
        self.transform = transform

        # 讀取 .txt 文件
        with open(txt_file, "r") as f:
            for line in f:
                path, label = line.strip().split()
                self.data.append((path, int(label)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]

        # 加載圖片
        image = Image.open(img_path).convert("RGB")  # 確保是 RGB 格式

        # 應用轉換
        if self.transform:
            image = self.transform(image)

        return image, label


# 訓練與驗證的資料增強與正規化
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),  # 隨機水平翻轉
    transforms.RandomRotation(degrees=15),  # 隨機旋轉 -15 到 15 度
    transforms.ColorJitter(brightness=0.2,
                           contrast=0.2,
                           saturation=0.2,
                           hue=0.1),  # 隨機調整顏色
    transforms.RandomResizedCrop(size=(224, 224),
                                 scale=(0.8, 1.0)),  # 隨機裁剪並調整大小
    transforms.ToTensor(),  # 轉換為 Tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 正規化
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 調整圖片大小
    transforms.ToTensor(),  # 轉換為 Tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 正規化
])

train_dataset = CustomDataset(train_txt, transform=train_transform)
val_dataset = CustomDataset(val_txt, transform=val_transform)

train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=4)

val_loader = DataLoader(val_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=4)

# 驗證修改
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# 3. 建立模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = models.resnet18(pretrained=True)  # 使用預訓練的 ResNet-18
# model.fc = nn.Linear(model.fc.in_features, num_classes)  # 調整最後的全連接層
# model = model.to(device)

# model = models.mobilenet_v2(pretrained=True)  # 使用預訓練的 MobileNetV2
# model.classifier[1] = nn.Linear(model.last_channel, num_classes)  # 修改分類器
# model = model.to(device)

model = models.efficientnet_b0(pretrained=True)  # 使用預訓練的 EfficientNet-B0
model.classifier[1] = nn.Linear(model.classifier[1].in_features,
                                num_classes)  # 修改分類器
model = model.to(device)

# 4. 定義損失函數與優化器
criterion = nn.CrossEntropyLoss()  # 交叉熵損失
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# 5. 訓練與驗證
print("Starting training...")

# 紀錄每個class最高的recall
best_recall = defaultdict(float)
besr_precision = defaultdict(float)
best_f1 = defaultdict(float)

for epoch in range(num_epochs):
    # 訓練階段
    model.train()
    running_train_loss = 0.0
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    train_loader_tqdm = tqdm(train_loader, desc="Training", leave=False)

    for inputs, labels in train_loader_tqdm:
        inputs, labels = inputs.to(device), labels.to(device)

        # 前向傳播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向傳播與優化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()

        # 更新進度條資訊
        train_loader_tqdm.set_postfix(loss=loss.item())

    train_loss = running_train_loss / len(train_loader)

    # 驗證階段
    model.eval()
    running_val_loss = 0.0
    correct = 0
    total = 0

    # 初始化每個分類的 TP, FP, FN 統計
    class_metrics = {
        i: {
            "TP": 0,
            "FP": 0,
            "FN": 0
        }
        for i in range(num_classes)
    }

    val_loader_tqdm = tqdm(val_loader, desc="Validating", leave=False)
    with torch.no_grad():
        for inputs, labels in val_loader_tqdm:
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向傳播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_val_loss += loss.item()

            # 計算正確率
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # 計算 TP, FP, FN
            for i in range(len(labels)):
                true_label = labels[i].item()
                pred_label = predicted[i].item()

                if true_label == pred_label:
                    class_metrics[true_label]["TP"] += 1
                else:
                    class_metrics[true_label]["FN"] += 1
                    class_metrics[pred_label]["FP"] += 1

    val_loss = running_val_loss / len(val_loader)
    val_accuracy = correct / total

    print(
        f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}"
    )

    # 計算每個分類的 Precision 和 Recall
    for class_id, metrics in class_metrics.items():
        TP = metrics["TP"]
        FP = metrics["FP"]
        FN = metrics["FN"]

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = (2 * precision * recall) / (precision + recall) if (
            precision + recall) > 0 else 0

        print(
            f"Class {class_id}: Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1_score:.4f}, TP: {TP}, FP: {FP}, FN: {FN}"
        )

        # if recall > best_recall[class_id]:
        #     best_recall[class_id] = recall
        #     torch.save(model.state_dict(),
        #                f"best_recall_classifier_model_{class_id}.pth")

        # if precision > besr_precision[class_id]:
        #     besr_precision[class_id] = precision
        #     torch.save(model.state_dict(),
        #                f"best_precision_classifier_model_{class_id}.pth")

        if f1_score > best_f1[class_id]:
            best_f1[class_id] = f1_score
            torch.save(model.state_dict(),
                       f"best_f1_classifier_model_{class_id}.pth")

# print("Best recall for each class:")
# for class_id, recall in best_recall.items():
#     print(f"Class {class_id}: {recall:.4f}")

# print("Best precision for each class:")
# for class_id, precision in besr_precision.items():
#     print(f"Class {class_id}: {precision:.4f}")

print("Best F1-score for each class:")
for class_id, f1 in best_f1.items():
    print(f"Class {class_id}: {f1:.4f}")

print("Training completed.")
