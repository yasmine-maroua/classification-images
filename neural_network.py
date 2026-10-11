import torch as th
import torch.optim as optim
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

#  ----------- 3.3)  Autres modélisations plus avancées ---------------
#  -------- réseau de neurones avec des couches linéaires


np.random.seed(0)

class NeuralNetwork(th.nn.Module):

    def __init__(self, d, k, h1=512, h2=256,h3=128):
        super(NeuralNetwork, self).__init__()

        # self.layer1 = th.nn.Linear(d, h1)
        # self.layer2 = th.nn.Linear(h1, h2)
        # self.layer3 = th.nn.Linear(h2, k)
        self.layer1 = th.nn.Linear(d, h1)
        self.layer2 = th.nn.Linear(h1, h2)
        self.layer3 = th.nn.Linear(h2, h3)
        self.layer4 = th.nn.Linear(h3, k)

        self.layer1.reset_parameters()
        self.layer2.reset_parameters()
        self.layer3.reset_parameters()
        self.layer4.reset_parameters()

    def forward(self, x):
        # phi1 = F.relu(self.layer1(x))
        # phi2 = F.relu(self.layer2(phi1))
        # return self.layer3(phi2)
        phi1 = F.relu(self.layer1(x))
        phi2 = F.relu(self.layer2(phi1))
        phi3 = F.relu(self.layer3(phi2))
        return self.layer4(phi3)


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


# Entrainement
def train_model(X_train, Y_train, X_valid, Y_valid, lr=0.01, nb_epochs=5000):
    device = "cpu"

    d = X_train.shape[1]
    k = len(np.unique(Y_train))

    model = NeuralNetwork(d, k)
    model = model.to(device)

    X_train = th.from_numpy(X_train).float().to(device)
    Y_train = th.from_numpy(Y_train).long().to(device)

    X_valid = th.from_numpy(X_valid).float().to(device)
    Y_valid = th.from_numpy(Y_valid).long().to(device)

    train_dataset = MyDataset(X_train, Y_train)
    train_loader = th.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

    criterion = th.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    pbar = tqdm(range(nb_epochs))
    for epoch in pbar:
        for batch_X, batch_Y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)
            loss.backward()
            optimizer.step()

        # évaluation toutes les epochs
        if epoch % 10 == 0:
            with th.no_grad():
                f_train = model(X_train)
                y_pred_train = prediction(f_train)
                acc_train = (y_pred_train == Y_train).float().mean()

                f_valid = model(X_valid)
                y_pred_valid = prediction(f_valid)
                acc_valid = (y_pred_valid == Y_valid).float().mean()

                pbar.set_postfix(
                    loss=loss.item(),
                    acc_train=acc_train.item(),
                    acc_valid=acc_valid.item()
                )

    return model


# Prediction sur test
def predict_test(model, X_test):
    device = "cpu"
    X_test = th.from_numpy(X_test).float().to(device)
    f = model(X_test)
    preds = prediction(f)
    return preds.cpu().numpy()


def save_predictions(predictions):
    np.savetxt("images_test_predictions.csv", predictions, fmt="%.18e")


with open("dataset_images_train", 'rb') as fo:
    dict_train = pickle.load(fo, encoding='bytes')

data_X = dict_train["data"]
data_Y = dict_train["target"]

with open("data_images_test", "rb") as fo:
    dict_test = pickle.load(fo)

X_test = dict_test["data"]

X_train, X_valid, Y_train, Y_valid = train_test_split(
    data_X, data_Y, test_size=0.2, random_state=42
)


mean = X_train.mean(axis=0)
std = X_train.std(axis=0) + 1e-8

X_train = (X_train - mean) / std
X_valid = (X_valid - mean) / std
X_test = (X_test - mean) / std

model = train_model( X_train, Y_train, X_valid, Y_valid, lr=0.0005, nb_epochs=150)


# ------- évaluation sur validation
X_valid_t = th.from_numpy(X_valid).float()
Y_valid_t = th.from_numpy(Y_valid).long()
outputs = model(X_valid_t)
preds = th.argmax(outputs, 1)
accuracy = (preds == Y_valid_t).sum().item() / Y_valid.shape[0]
print("Validation accuracy :", accuracy)

# ------- prédictions sur le dataset test
predictions = predict_test(model, X_test)
save_predictions(predictions)