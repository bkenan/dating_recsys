import time
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from model import NNColabFiltering


def main():
    """
    Performs model training.
    Output: The saved model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ratings = pd.read_csv('./data/ratings.csv')
    ratings['ratings'] = ratings['ratings'].astype(float)
    ratings[ratings < 0] = 0
    X = ratings.loc[:, ['userID', 'itemID']]
    y = ratings.loc[:, 'ratings']

    # Hyperparameters:
    batchsize = 64
    criterion = nn.MSELoss()
    lr = 0.001
    n_epochs = 10
    wd = 1e-3

    # Split our data into training and test sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, random_state=0, test_size=0.2)

    def prep_dataloaders(X_train, y_train, X_val, y_val, batch_size):

        # Convert training and test data to TensorDatasets
        trainset = TensorDataset(torch.from_numpy(np.array(X_train)).long(),
                                 torch.from_numpy(np.array(y_train)).float())
        valset = TensorDataset(torch.from_numpy(np.array(X_val)).long(),
                               torch.from_numpy(np.array(y_val)).float())

        # Create Dataloaders for our training and test data to allow us
        # to iterate over minibatches
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True)
        valloader = torch.utils.data.DataLoader(
            valset, batch_size=batch_size, shuffle=False)

        return trainloader, valloader

    trainloader, valloader = prep_dataloaders(
        X_train, y_train, X_val, y_val, batchsize)

    def train_model(model, criterion, optimizer, dataloaders,
                    device, num_epochs=5, scheduler=None):

        model = model.to(device)  # Send model to GPU if available
        since = time.time()

        costpaths = {'train': [], 'val': []}

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0

                # Get the inputs and labels, and send to GPU if available
                for (inputs, labels) in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # Zero the weight gradients
                    optimizer.zero_grad()

                    # Forward pass to get outputs and calculate loss
                    # Track gradient only for training data
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model.forward(inputs).view(-1)
                        loss = criterion(outputs, labels)

                        # Backpropagation to get the gradients with respect
                        # to each weight
                        # Only if in train
                        if phase == 'train':
                            loss.backward()
                            # Update the weights
                            optimizer.step()

                    # Convert loss into a scalar and add it to running_loss
                    running_loss += np.sqrt(loss.item()) * labels.size(0)

                # Step along learning rate scheduler when in train
                if (phase == 'train') and (scheduler is not None):
                    scheduler.step()

                # Calculate and display average loss and accuracy for the epoch
                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                costpaths[phase].append(epoch_loss)
                print('{} loss: {:.4f}'.format(phase, epoch_loss))

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

        return costpaths

    # Train the model
    dataloaders = {'train': trainloader, 'val': valloader}
    n_users = X.loc[:, 'userID'].max()+1
    n_items = X.loc[:, 'itemID'].max()+1
    model = NNColabFiltering(n_users, n_items, embedding_dim_users=50,
                             embedding_dim_items=50, n_activations=100,
                             rating_range=[0., 6.])
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    cost_paths = train_model(model, criterion, optimizer,
                             dataloaders, device, n_epochs, scheduler=None)
    print(cost_paths)

    # Saving the model
    PATH = "./model.pt"
    torch.save(model.state_dict(), PATH)


if __name__ == '__main__':
    main()
