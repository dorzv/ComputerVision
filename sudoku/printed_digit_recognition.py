from glob import glob
import random
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, PILToTensor
from torch.optim import SGD, Adam
import numpy as np
import matplotlib.pyplot as plt
device = "cuda" if torch.cuda.is_available() else "cpu"


class PrintedDigits(Dataset):
    def __init__(self, n, random_sate=42, transform=None):
        self.n = n
        self.random_state = random_sate
        self.transform = transform

        fonts_folder = 'data/fonts'
        self.fonts = glob(fonts_folder + '/*.ttf')

        # random.seed(random_sate)

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        target = random.randint(1, 9)
        size = random.randint(230, 280)
        x = random.randint(30, 90)
        y = random.randint(30, 90)
        color = 255

        image = Image.new("L", (256, 256))
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(random.choice(self.fonts), size)
        draw.text((x, y), str(target), color, font=font, anchor='lt')
        image = image.resize((28, 28), Image.BILINEAR)

        if self.transform:
            image = self.transform(image)
            image = image.float()/255

        return image.to(device), torch.tensor([target]).to(device)



def get_model():
    model = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(6400, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10)
        ).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)
    return model, loss_fn, optimizer


def train_batch(x, y, model, opt, loss_fn):
    prediction = model(x)
    batch_loss = loss_fn(prediction, y)
    batch_loss.backward()
    opt.step()
    opt.zero_grad()
    return batch_loss.item()


@torch.no_grad()
def accuracy(x, y, model):
    model.eval()
    prediction = model(x)
    max_values, argmaxes = prediction.max(-1)
    is_correct = argmaxes == y
    return is_correct.cpu().numpy().tolist()


def get_data():
    train = PrintedDigits(50_000, transform=PILToTensor())
    trn_dl = DataLoader(train, batch_size=2)
    val = PrintedDigits(10_000, random_sate=3, transform=PILToTensor())
    val_dl = DataLoader(val, batch_size=32)
    return trn_dl, val_dl


@torch.no_grad()
def val_loss(x, y, model):
    model.eval()
    prediction = model(x)
    val_loss = loss_fn(prediction, y)
    return val_loss.item()


if __name__ == '__main__':
    trn_dl, val_dl = get_data()
    model, loss_fn, optimizer = get_model()

    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    for epoch in range(5):
        print(epoch)
        train_epoch_losses, train_epoch_accuracies = [], []
        print('Training...')
        random.seed(42)
        for ix, batch in enumerate(iter(trn_dl)):
            x, y = batch
            y = y.squeeze()
            batch_loss = train_batch(x, y, model, optimizer, loss_fn)
            train_epoch_losses.append(batch_loss)
        train_epoch_loss = np.array(train_epoch_losses).mean()

        print('Validate on training...')
        random.seed(42)
        for ix, batch in enumerate(iter(trn_dl)):
            x, y = batch
            y = y.squeeze()
            is_correct = accuracy(x, y, model)
            train_epoch_accuracies.extend(is_correct)
        train_epoch_accuracy = np.mean(train_epoch_accuracies)

        print('Validate on validation...')
        random.seed(3)
        for ix, batch in enumerate(iter(val_dl)):
            x, y = batch
            y = y.squeeze()
            val_is_correct = accuracy(x, y, model)
            validation_loss = val_loss(x, y, model)
        val_epoch_accuracy = np.mean(val_is_correct)

        train_losses.append(train_epoch_loss)
        train_accuracies.append(train_epoch_accuracy)
        val_losses.append(validation_loss)
        val_accuracies.append(val_epoch_accuracy)

    epochs = np.arange(5)+1
    plt.subplot(211)
    plt.plot(epochs, train_losses, 'bo', label='Training loss')
    plt.plot(epochs, val_losses, 'r', label='Validation loss')
    # plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.title('Training and validation loss with CNN')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid('off')
    plt.show()
    plt.subplot(212)
    plt.plot(epochs, train_accuracies, 'bo', label='Training accuracy')
    plt.plot(epochs, val_accuracies, 'r', label='Validation accuracy')
    # plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.title('Training and validation accuracy with CNN')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid('off')
    plt.show()