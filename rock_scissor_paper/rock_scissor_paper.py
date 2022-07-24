# https://learnopencv.com/playing-rock-paper-scissors-with-ai/
# https://github.com/spmallick/learnopencv/blob/master/Playing-rock-paper-scissors-with-AI/Rock_paper_scissor.ipynb

import os
from time import sleep
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch import nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
device = 'cuda' if torch.cuda.is_available() else 'cpu'

show_plots = True
user_paper = cv2.imread('user_paper.png')
user_scissor = cv2.imread('user_scissor.png')
user_rock = cv2.imread('user_rock.png')
computer_paper = cv2.imread('computer_paper.png')
computer_scissor = cv2.imread('computer_scissor.png')
computer_rock = cv2.imread('computer_rock.png')

user_images = [user_paper, user_rock, user_scissor]
computer_images = [computer_paper, computer_rock, computer_scissor]

for ii in range(3):
    user_images[ii] = cv2.resize(user_images[ii], (70, 70))
    computer_images[ii] = cv2.resize(computer_images[ii], (70, 70))


def generate_data(num_of_sample: int):
    global rock, paper, scissor, nothing

    capture = cv2.VideoCapture(0)
    trigger = False
    counter = 0
    box_size = 234
    height = int(capture.get(4))
    width = int(capture.get(3))


    while True:
        ret, frame = capture.read()
        frame = cv2.flip(frame, 1)
        if not ret:
            break

        if counter == num_of_sample:
            trigger = False
            counter = 0

        cv2.rectangle(frame, (width - box_size, 0), (width, box_size), (0, 255, 0), 2)
        cv2.namedWindow("Collect data", cv2.WINDOW_NORMAL)

        if trigger:
            roi = frame[5:box_size-5, width-box_size+5:width-5]
            current_list.append([roi, class_name])
            counter += 1
            text = f'Collecting Sample for "{class_name}" class: {counter}'

        else:
            text = 'Press "p" for paper, "s" for scissor, "r" for rock or "n" for nothing'

        cv2.putText(frame, text, (3, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, 'Press "q" to continue', (3, height - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow("Collect data", frame)

        k = cv2.waitKey(1)
        if k == ord('p'):
            trigger = True
            class_name = 'paper'
            current_list = paper
        elif k == ord('s'):
            trigger = True
            class_name = 'scissor'
            current_list = scissor
        elif k == ord('r'):
            trigger = True
            class_name = 'rock'
            current_list = rock
        elif k == ord('n'):
            trigger = True
            class_name = 'nothing'
            current_list = nothing
        elif k == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()


rock = []
paper = []
scissor = []
nothing = []
show_data = False
num_of_samples = 100
generate_data(num_of_samples)

if show_data:
    rows = 4
    cols = 8
    fig, ax = plt.subplots(rows, cols)
    for class_items, plot_row in zip([rock, paper, scissor, nothing], ax):
        for plot_cell in plot_row:
            plot_cell.grid(False)
            plot_cell.axis('off')
            image, _ = random.choice(class_items)
            plot_cell.imshow(image[:, :, ::-1])

# preprocessing
labels = len(rock) * ['rock'] + len(paper) * ['paper'] + len(scissor) * ['scissor'] + len(nothing) * ['nothing']
images = [x[0][:, :, ::-1] for x in rock] + [x[0][:, :, ::-1] for x in paper] + [x[0][:, :, ::-1] for x in scissor] + [x[0][:, :, ::-1] for x in nothing]
images = np.array(images, dtype=float) / 255
encoder = LabelEncoder()
integer_labels = encoder.fit_transform(labels)  # nothing = 0 paper = 1, rock = 2, scissor = 3

train_images, test_images, train_labels, test_labels = train_test_split(images, integer_labels, test_size=0.25,
                                                                        random_state=42)
del images

# prepare model
model = models.efficientnet_b0(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

model.classifier = nn.Sequential(nn.Dropout(0.2),
                                 nn.Linear(1280, 4))
model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


class RSPDataset(Dataset):
    def __init__(self, x, y):
        x = torch.tensor(x)
        x = x.float()
        x = x.view(-1, 3, 224, 224)
        y = torch.tensor(y)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.x = x
        self.y = y

    def __getitem__(self, ix):
        x, y = self.x[ix], self.y[ix]
        x = self.normalize(x)
        return x.to(device), y.to(device)

    def __len__(self):
        return len(self.x)


def train_batch(x, y, model, opt, loss_fn):
    model.train()
    prediction = model(x)
    batch_loss = loss_fn(prediction, y)
    batch_loss.backward()
    opt.step()
    opt.zero_grad()
    return batch_loss.item()


@torch.no_grad()
def val_loss(x, y, model, loss_fn):
    model.eval()
    prediction = model(x)
    val_loss = loss_fn(prediction, y)
    return val_loss.item()


@torch.no_grad()
def accuracy(x, y, model):
    model.eval()
    prediction = model(x)
    max_values, argmaxes = prediction.max(-1)
    is_correct = y == argmaxes
    return is_correct.cpu().numpy().tolist()


train = RSPDataset(train_images, train_labels)
trn_dl = DataLoader(train, batch_size=20, shuffle=True, drop_last=True)
val = RSPDataset(test_images, test_labels)
val_dl = DataLoader(val, batch_size=20, shuffle=True, drop_last=True)
epochs = 6

train_losses, train_accuracies = [], []
val_losses, val_accuracies = [], []
for epoch in range(epochs):
    print(epoch)
    train_epoch_losses, train_epoch_accuracies = [], []
    val_epoch_losses, val_epoch_accuracies = [], []
    for ix, batch in enumerate(iter(trn_dl)):
        x, y = batch
        batch_loss = train_batch(x, y, model, optimizer, loss_fn)
        train_epoch_losses.append(batch_loss)
    train_epoch_loss = np.array(train_epoch_losses).mean()

    for ix, batch in enumerate(iter(trn_dl)):
        x, y = batch
        is_correct = accuracy(x, y, model)
        train_epoch_accuracies.extend(is_correct)
    train_epoch_accuracy = np.mean(train_epoch_accuracies)

    for ix, batch in enumerate(iter(val_dl)):
        x, y = batch
        val_is_correct = accuracy(x, y, model)
        val_epoch_accuracies.extend(val_is_correct)
        validation_loss = val_loss(x, y, model, loss_fn)
        val_epoch_losses.append(validation_loss)
    val_epoch_accuracy = np.mean(val_epoch_accuracies)
    val_epoch_loss = np.array(val_epoch_losses).mean()

    train_losses.append(train_epoch_loss)
    train_accuracies.append(train_epoch_accuracy)
    val_losses.append(val_epoch_loss)
    val_accuracies.append(val_epoch_accuracy)

if show_plots:
    epochs = np.arange(epochs)+1
    plt.subplot(211)
    plt.semilogy(epochs, train_losses, 'bo', label='Training loss')
    plt.plot(epochs, val_losses, 'r', label='Validation loss')
    # plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.title('Training and validation loss with CNN')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid('off')
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

model.eval()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


# play the game
def play_game(model):
    capture = cv2.VideoCapture(0)
    box_size = 234
    height = int(capture.get(4))
    width = int(capture.get(3))

    while True:
        ret, frame = capture.read()
        frame = cv2.flip(frame, 1)
        if not ret:
            break

        cv2.rectangle(frame, (width - box_size, 0), (width, box_size), (0, 255, 0), 2)
        cv2.namedWindow("Play", cv2.WINDOW_NORMAL)

        text = 'Click p to play'
        cv2.putText(frame, text, (3, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow("Play", frame)

        k = cv2.waitKey(1)
        if k == ord('p'):
            roi = frame[5:box_size - 5, width - box_size + 5:width - 5]
            x = roi[:, :, ::-1] / 255
            x = torch.tensor(x).float()
            x = x.view(3, 224, 224)
            x = x.unsqueeze(0)
            x = normalize(x)
            x = x.to(device)
            prediction = model(x)
            max_values, argmaxes = prediction.max(-1)
            user_choise = argmaxes.item()
            if user_choise == 0:
                text = 'Nothing chose'
            else:
                computer_choice = random.randint(1, 3)
                computer_image = computer_images[computer_choice - 1]
                user_image = user_images[user_choise - 1]
                if computer_choice == user_choise:
                    text = "It's a tie"
                elif computer_choice == 1:  # paper
                    if user_choise == 3:  # scissor
                        text = 'You win!'
                    else:
                        text = 'Computer win'
                elif computer_choice == 2:  # rock
                    if user_choise == 1:
                        text = 'You win!'
                    else:
                        text = 'Computer win'
                else:  # computer chose scissor
                    if user_choise == 2:
                        text = 'You win!'
                    else:
                        text = 'Computer win'
            ret, frame = capture.read()
            frame = cv2.flip(frame, 1)
            cv2.rectangle(frame, (width - box_size, 0), (width, box_size), (0, 255, 0), 2)
            cv2.putText(frame, text, (width // 5 , height // 2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),
                        3, cv2.LINE_AA)
            if text != 'Nothing chose':
                frame[height - 120: height - 50, 10:80] = computer_image
                frame[height - 120: height - 50, width - 80:width - 10] = user_image
            cv2.imshow("Play", frame)
            cv2.waitKey(2000)

        elif k == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()


play_game(model)
