import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import pickle

'''
Homework: Image Classification using CNN
Instructions:
1. Follow the skeleton code precisely as provided.
2. You may define additional helper functions if necessary, but ensure the input/output format is maintained.
3. Visualize and report results as specified in the problem.
'''
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')
# (a) Dataloader: Download the MNIST dataset and get the dataloader in PyTorch
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms)
# Split the training set into training and validation sets
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [int(0.8 * len(train_dataset)), len(train_dataset) - int(0.8 * len(train_dataset))])
# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)

# (c) Modeling: Implement a Convolutional Neural Network
class cnn_block(nn.Module): # week 11 discussion code, but with 1 in_channel for our grayscale 28x28 images
  def __init__(self, in_channels, n_hidden, kernel_size, stride):
    super().__init__()
    self.layers = nn.Sequential(
        nn.Conv2d(in_channels = in_channels, out_channels = n_hidden, kernel_size = kernel_size, bias=False, padding = 'same', stride = stride),
        nn.BatchNorm2d(num_features = n_hidden),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Conv2d(in_channels = n_hidden, out_channels = in_channels, kernel_size = kernel_size, bias=False, padding = 'same', stride = stride),
        nn.BatchNorm2d(num_features = in_channels),
        nn.ReLU(),
        nn.Dropout(p=0.2))

    def forward(self, x):
        return x + self.layers(x)

class linear_block(nn.Module):
  def __init__(self, in_features, n_hidden):
    super().__init__()
    self.in_features = (in_features, n_hidden)
    self.layers = nn.Sequential(
        nn.Linear(in_features = in_features, out_features = n_hidden),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(in_features = n_hidden, out_features = in_features),
        nn.ReLU()
    )

    def forward(self, x):
        return x + self.layers(x)

class CNNModel(nn.Module):
    '''
    CNN Model for image classification.    
    '''
    def __init__(self, in_channels = 3, cnn_channels = 5, linear_hidden = 1000, n_classes = 10, kernel_size = (3, 3), stride = (1, 1)):
        super().__init__()
        # Define the CNN layers
        self.cnn_layers = nn.Sequential(
        cnn_block(in_channels, cnn_channels, kernel_size, stride=stride),
        cnn_block(in_channels, cnn_channels, kernel_size, stride=stride),
        cnn_block(in_channels, cnn_channels, kernel_size, stride=stride))

        self.down_sample = nn.Conv2d(in_channels = in_channels, out_channels = 1, kernel_size = (1, 1))

        self.linear_layers = nn.Sequential(
            linear_block(28*28, linear_hidden),
            linear_block(28*28, linear_hidden)
        )
        self.last_layer = nn.Linear(28*28, n_classes)

        self.all = nn.Sequential(
            self.cnn_layers,
            self.down_sample,
            nn.Flatten(),
            self.linear_layers,
            self.last_layer,
        )

        def forward(self, x):
            return self.all(x)

# (1) Training and Evaluation
# Students should define the optimizer and scheduler outside this function if needed
def train_and_evaluate(model, train_loader, val_loader, epochs=60):
    '''
    Train the model and evaluate on the validation set.

    Parameters:
        model (CNNModel): The CNN model to train.
        train_loader (DataLoader): The training data loader.
        val_loader (DataLoader): The validation data loader.
        epochs (int): Number of training epochs.
    Returns:
        float: Validation accuracy.
    '''
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    train_loss, test_loss = [], []
    for epoch in range(epochs):
        # do a loop over all training samples
        model.train() # telling the model we are training it as it needs to keep track of gradients (and other things) in this modality
        epoch_train_loss = []
        for batch, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            preds = model(X)
            loss = nn.CrossEntropyLoss(preds, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss.append(loss.item())
        # do a loop over all testing samples
        model.eval() # telling the model we are evaluating it
        epoch_test_loss = []
        with torch.no_grad(): # alternatively, torch.no_grad()
            for batch, (X, y) in enumerate(val_loader):
                X, y = X.to(device), y.to(device)
                preds = model(X)
                loss  = nn.CrossEntropyLoss(preds, y)
                epoch_test_loss.append(loss.item())
        train_loss.append(np.mean(epoch_train_loss))
        test_loss.append(np.mean(epoch_test_loss))
    return model, np.array(test_loss[-3]).mean()

# (4) Hyper-parameter tuning function
# This function should not use excessive resources and should be efficient for a sandbox environment
def tune_hyperparameters(train_loader, val_loader):
    '''
    Tune hyperparameters of the CNN model using the validation set.

    Parameters:
        train_loader (DataLoader): The training data loader.
        val_loader (DataLoader): The validation data loader.
    Returns:
        CNNModel: The best model.
    '''
    best_model = CNNModel(cnn_channels=10, linear_hidden=10)  # Placeholder for the best model (my homework answer)
    best_score = train_and_evaluate(best_model, train_loader, val_loader, 60)  # Placeholder for the best score

    # my homework file tuned kernel and stride under various cnn channels
    # I wan't to see if increasing nn depth will improve accuracy
    # this will test 18 models changing cnn and linear depth
    # since a larger network will require more training, I also increase n_epochs
    for cnn_channels in [10,12,15]:
        for linear_hidden in [10,15]:
            for epochs in [70,80]:
                model = CNNModel(cnn_channels=cnn_channels, linear_hidden=linear_hidden)
                model, score = train_and_evaluate(model, train_loader, val_loader, epochs)
                if score>best_score:
                    best_model = model
                    best_score = score

    return best_model

if __name__ == "__main__":
    # Obtain train_loader and val_loader here
    # train_loader, val_loader = DataLoader definitions outside this code block
    
    # Tune hyperparameters and find the best model
    best_model = tune_hyperparameters(train_loader, val_loader)

    # Save the best model to a pickle file
    with open('best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
