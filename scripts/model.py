import torch
import torch.nn.functional as F
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NNColabFiltering(nn.Module):

    def __init__(self, n_users, n_items, embedding_dim_users, embedding_dim_items, n_activations, rating_range):
        super().__init__()
        self.user_embeddings = nn.Embedding(
            num_embeddings=n_users, embedding_dim=embedding_dim_users)
        self.item_embeddings = nn.Embedding(
            num_embeddings=n_items, embedding_dim=embedding_dim_items)
        self.fc1 = nn.Linear(embedding_dim_users +
                             embedding_dim_items, n_activations)
        self.fc2 = nn.Linear(n_activations, 1)
        self.rating_range = rating_range

    def forward(self, X):
        # Get embeddings for minibatch
        embedded_users = self.user_embeddings(X[:, 0])
        embedded_items = self.item_embeddings(X[:, 1])
        # Concatenate user and item embeddings
        embeddings = torch.cat([embedded_users, embedded_items], dim=1)
        # Pass embeddings through network
        preds = self.fc1(embeddings)
        preds = F.relu(preds)
        preds = self.fc2(preds)
        # Scale predicted ratings to target-range [low,high]
        preds = torch.sigmoid(
            preds) * (self.rating_range[1]-self.rating_range[0]) + self.rating_range[0]
        return preds
