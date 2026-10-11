import torch as th
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split


device = th.device("cuda" if th.cuda.is_available() else "cpu")


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = th.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return th.relu(out)


class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = th.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        return self.linear(out)



stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))


train_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])

test_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])


class MyDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx].reshape(3, 32, 32).transpose(1, 2, 0)
        return self.transform(x), self.labels[idx]


def train():
    print("Chargement des données...")
    with open("dataset_images_train", 'rb') as f:
        d = pickle.load(f)
    X_tr, X_va, Y_tr, Y_va = train_test_split(d["data"], d["target"], test_size=0.05, random_state=42)

    train_loader = DataLoader(MyDataset(X_tr, Y_tr, train_tf), batch_size=128, shuffle=True, num_workers=2)
    valid_loader = DataLoader(MyDataset(X_va, Y_va, test_tf), batch_size=128, num_workers=2)

    model = ResNet18().to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc = 0
    print("Début de l'entraînement :")

    for epoch in range(200):
        model.train()
        train_loss = 0
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch}/200", leave=False):
            xb, yb = xb.to(device), yb.to(device).long()
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()


        model.eval()
        correct, total = 0, 0
        with th.no_grad():
            for xb, yb in valid_loader:
                xb, yb = xb.to(device), yb.to(device)
                correct += (th.argmax(model(xb), 1) == yb).sum().item()
                total += yb.size(0)

        acc = correct / total
        print(
            f"Epoch {epoch} | Loss: {train_loss / len(train_loader):.3f} | Acc Valid: {acc:.4f} | LR: {scheduler.get_last_lr()[0]:.5f}")

        if acc > best_acc:
            best_acc = acc
            th.save(model.state_dict(), 'best_resnet18.pth')

    return model


def save_preds_tta(model):
    with open("data_images_test", "rb") as f:
        dtest = pickle.load(f)

    tta_transforms = [
        test_tf,
        transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(p=1), transforms.ToTensor(),
                            transforms.Normalize(*stats)]),
        transforms.Compose([transforms.ToPILImage(), transforms.ColorJitter(brightness=0.1), transforms.ToTensor(),
                            transforms.Normalize(*stats)])
    ]

    model.eval()
    all_probs = []

    with th.no_grad():
        for t in tta_transforms:
            test_ds = MyDataset(dtest["data"], np.zeros(len(dtest["data"])), t)
            loader = DataLoader(test_ds, batch_size=128, shuffle=False)
            outputs = []
            for xb, _ in loader:
                outputs.append(th.softmax(model(xb.to(device)), dim=1))
            all_probs.append(th.cat(outputs))

    final_avg = th.stack(all_probs).mean(0)
    preds = th.argmax(final_avg, 1).cpu().numpy()

    np.savetxt("images_test_predictions.csv", preds, fmt="%.18e")



if __name__ == "__main__":
    trained_model = train()
    save_preds_tta(trained_model)