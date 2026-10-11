import torch as th
import torch.optim as optim
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

#  ----------- 3.3)  Autres modélisations plus avancées ---------------
#  --------  réseau de convolution (CNN)

np.random.seed(0)


class CNN(th.nn.Module):

    def __init__(self, k):
        super(CNN, self).__init__()

        # Convolutions
        self.conv1 = th.nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = th.nn.BatchNorm2d(32)

        self.conv2 = th.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = th.nn.BatchNorm2d(64)

        self.conv3 = th.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = th.nn.BatchNorm2d(128)

        self.pool = th.nn.MaxPool2d(2, 2)


        self.fc1 = th.nn.Linear(128 * 4 * 4, 256)
        self.dropout = th.nn.Dropout(0.5)
        self.fc2 = th.nn.Linear(256, k)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 32→16
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 16→8
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 8→4

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        return self.fc2(x)


def prediction(f):
    return th.argmax(f, 1)



class MyDataset(th.utils.data.Dataset):

    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]



def train_model(X_train, Y_train, X_valid, Y_valid,
                lr=0.0003, nb_epochs=40, batch_size=64):

    device = "cpu"
    k = len(np.unique(Y_train))

    model = CNN(k).to(device)

    X_train = X_train.reshape(-1, 3, 32, 32)
    X_valid = X_valid.reshape(-1, 3, 32, 32)

    X_train = th.from_numpy(X_train).float()
    Y_train = th.from_numpy(Y_train).long()

    X_valid = th.from_numpy(X_valid).float()
    Y_valid = th.from_numpy(Y_valid).long()

    train_dataset = MyDataset(X_train, Y_train)
    train_loader = th.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    criterion = th.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = th.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    pbar = tqdm(range(nb_epochs))

    for epoch in pbar:
        model.train()

        for batch_X, batch_Y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)
            loss.backward()
            optimizer.step()

        scheduler.step()

        model.eval()
        with th.no_grad():
            outputs = model(X_valid)
            preds = prediction(outputs)
            acc = (preds == Y_valid).float().mean()

        pbar.set_postfix(loss=loss.item(), acc_valid=acc.item())

    return model


def predict_test(model, X_test):
    model.eval()

    X_test = X_test.reshape(-1, 3, 32, 32)
    X_test = th.from_numpy(X_test).float()

    with th.no_grad():
        outputs = model(X_test)
        preds = prediction(outputs)

    return preds.numpy()


def save_predictions(predictions):
    np.savetxt("images_test_predictions.csv", predictions, fmt="%d")


with open("dataset_images_train", 'rb') as fo:
    dict_train = pickle.load(fo)

data_X = dict_train["data"]
data_Y = dict_train["target"]

with open("data_images_test", "rb") as fo:
    dict_test = pickle.load(fo)

X_test = dict_test["data"]


X_train, X_valid, Y_train, Y_valid = train_test_split(data_X, data_Y, test_size=0.2, random_state=42)


mean = X_train.mean(axis=0)
std = X_train.std(axis=0) + 1e-8

X_train = (X_train - mean) / std
X_valid = (X_valid - mean) / std
X_test = (X_test - mean) / std


model = train_model(
    X_train, Y_train,
    X_valid, Y_valid,
    lr=0.0002,
    nb_epochs=60,
    batch_size=32
)


X_valid_t = th.from_numpy(X_valid.reshape(-1, 3, 32, 32)).float()
Y_valid_t = th.from_numpy(Y_valid).long()

model.eval()
with th.no_grad():
    outputs = model(X_valid_t)
    preds = th.argmax(outputs, 1)

accuracy = (preds == Y_valid_t).sum().item() / Y_valid.shape[0]
print("Validation accuracy :", accuracy)


predictions = predict_test(model, X_test)
save_predictions(predictions)
