import os
import torch
import torch.utils.data
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import re
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

class PedDataset(torch.utils.data.Dataset): 
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.img_dir = os.path.join(root, "PNGImages")
        self.ann_dir = os.path.join(root, "Annotation")
        self.imgs = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.png')])
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.imgs[index])
        ann_filename = os.path.splitext(self.imgs[index])[0] + '.txt'
        ann_path = os.path.join(self.ann_dir, ann_filename)

        img = Image.open(img_path).convert("RGB")

        with open(ann_path, 'r') as f:
            ann_text = f.read()
        pattern = r"Bounding box for object \d+ \"[^\"]+\" \(Xmin, Ymin\) - \(Xmax, Ymax\) : \((\d+), (\d+)\) - \((\d+), (\d+)\)"
        raw_boxes = re.findall(pattern, ann_text)

        boxes = []
        for box in raw_boxes:
            coords = [float(c) for c in box]
            boxes.append(coords)
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.ones((len(boxes),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([index]),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if boxes.numel() > 0 else torch.tensor(0),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64)
        }

        img = F.to_tensor(img)

        if self.transforms is not None:
            pass
        
        return img, target
    def __len__(self):
        return len(self.imgs)

def our_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_f = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_f, num_classes)
    return model

def group_correctly(batch):
    return tuple(zip(*batch))

def evaluate_model(model, data_loader, device):
    model.eval()
    metrics = MeanAveragePrecision()
    with torch.no_grad():
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            targets = [{k:v.to(device) for k, v in t.items()} for t in targets]
            predictions = model(images)
            metrics.update(predictions, targets)
    results = metrics.compute()
    for k, v in results.items():
        print(f"{k}: {v.item():.4f}")

    return results

def visualize_predictions(model, dataset, device, n_imgs = 5):
    model.eval()
    all_indices = list(range(len(dataset)))
    random_indices = random.sample(all_indices, n_imgs)


    fig, axes = plt.subplots(1, n_imgs, figsize=(20, 5))
    fig.suptitle("Model Predictions on Random Test Images", fontsize=16)

    for i, idx in enumerate(random_indices):
        img_tensor, _ = dataset[idx]
        with torch.no_grad():
            prediction = model([img_tensor.to(device)])[0]
        img_display = img_tensor.permute(1, 2, 0).numpy()
        ax = axes[i]
        ax.imshow(img_display)
        ax.set_title(f"Image #{idx}")
        ax.axis('off')
        for box, score, label in zip(prediction['boxes'], prediction['scores'], prediction['labels']):
            if score > 0.5:
                box = box.cpu().numpy()
                xmin, ymin, xmax, ymax = box
                rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                         linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                ax.text(xmin, ymin, f'Pedestrian: {score:.2f}',
                        bbox=dict(facecolor='yellow', alpha=0.5))
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def main():
    root = "dataset"
    if not os.path.exists(os.path.join(root, "PNGImages")):
        print(f"Error")
        return
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    dataset = PedDataset(root = root)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    dataset_train, dataset_test = torch.utils.data.random_split(dataset, [train_size, test_size])

    print(f"Training set size: {len(dataset_train)}")
    print(f"Test set size: {len(dataset_test)}")

    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=4, num_workers=2, shuffle=True, collate_fn=group_correctly)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, num_workers=2, shuffle=False, collate_fn=group_correctly)

    num_classes = 2
    model = our_model(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for i, (images, targets) in enumerate(dataloader_train):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            #in these three steps the main learning happens in the model using backpropogation
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()
        lr_scheduler.step()
        print(f"Epoch {epoch + 1}: \nAverage Loss: {epoch_loss / len(dataloader_train):.4f}")

    print("Done!!!")
    torch.save(model.state_dict(), "ped-detection-model.pth")

    print("\n--- Evaluating on Test Set ---")
    evaluate_model(model, dataloader_test, device)
    
    print("\n--- Visualizing Predictions ---")
    visualize_predictions(model, dataset_test, device)

if __name__ == "__main__":
    main()