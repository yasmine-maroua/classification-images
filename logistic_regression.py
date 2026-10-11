import numpy as np
import torch as th
import pickle
import torch.optim as optim
from torch.nn import functional as F
from tqdm import tqdm

#  ----------- 3.3)  Autres modélisations plus avancées ---------------
#  -------- algorithme de régression logisitique multivariée

class RegLogMulti(th.nn.Module):

    def __init__(self, d, k):
        super(RegLogMulti, self).__init__()
        self.layer = th.nn.Linear(d, k)
        self.layer.reset_parameters()

    def forward(self, x):
        out = self.layer(x)
        return F.softmax(out, dim=1)


def prediction(f):
    return th.argmax(f, 1)


def train_model(X_train, y_train, nb_epochs=2000, lr=0.01):

    device = "cpu"

    d = X_train.shape[1]
    k = len(np.unique(y_train))

    model = RegLogMulti(d, k).to(device)

    X_train = th.from_numpy(X_train).float().to(device)
    y_train = th.from_numpy(y_train).long().to(device)

    criterion = th.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    pbar = tqdm(range(nb_epochs))

    for i in pbar:
        optimizer.zero_grad()
        f = model(X_train)
        loss = criterion(f, y_train)
        loss.backward()
        optimizer.step()
        if i % 200 == 0:
            y_pred = prediction(f)
            acc = (y_pred == y_train).sum().item() / y_train.size(0)
            pbar.set_postfix(loss=loss.item(), accuracy=acc)
    return model


def predict_test(model, X_test):
    device = "cpu"
    X_test = th.from_numpy(X_test).float().to(device)
    f = model(X_test)
    y_pred = prediction(f)
    return y_pred.detach().cpu().numpy()


def save_predictions(predictions, filename="images_test_predictions.csv"):
    np.savetxt(
        filename,
        predictions,
        fmt="%.18e"
    )

with open("dataset_images_train", 'rb') as fo:
    dict_train = pickle.load(fo, encoding='bytes')

data_X = dict_train["data"]
data_Y = dict_train["target"]

with open("data_images_test", "rb") as fo:
    dict_test = pickle.load(fo)

X_test = dict_test["data"]

n = data_X.shape[0]
indices = np.random.permutation(n)
train_size = int(0.8 * n)

train_idx = indices[:train_size]
valid_idx = indices[train_size:]

X_train = data_X[train_idx]
Y_train = data_Y[train_idx]

X_valid = data_X[valid_idx]
Y_valid = data_Y[valid_idx]

# print("train:", X_train.shape)
# print("validation:", X_valid.shape)

mean = X_train.mean(axis=0)
std = X_train.std(axis=0) + 1e-8

X_train = (X_train - mean) / std
X_valid = (X_valid - mean) / std
X_test = (X_test - mean) / std

model = train_model(X_train, Y_train, nb_epochs=100, lr=0.001)

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