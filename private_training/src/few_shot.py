import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

class PrototypicalNetwork(nn.Module):
    """
    Implementation of Prototypical Networks for Few-Shot Learning
    (Snell et al., 2017).
    """
    def __init__(self, backbone, feature_dim):
        super(PrototypicalNetwork, self).__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        
    def forward(self, support_set, query_set, n_way, n_shot):
        """
        Perform few-shot classification
        
        Args:
            support_set: Support set images [n_way * n_shot, channels, height, width]
            query_set: Query set images [n_query, channels, height, width]
            n_way: Number of classes in the task
            n_shot: Number of examples per class in the support set
            
        Returns:
            Query set predictions
        """
        # Extract features
        support_features = self.backbone(support_set)  # [n_way * n_shot, feature_dim]
        query_features = self.backbone(query_set)      # [n_query, feature_dim]
        
        # Compute prototypes
        support_features = support_features.view(n_way, n_shot, -1).mean(dim=1)  # [n_way, feature_dim]
        
        # Compute distances
        dists = torch.cdist(query_features, support_features)  # [n_query, n_way]
        
        # Return negative distances (higher means closer)
        return -dists

class EpisodicDataset(Dataset):
    """
    Dataset for episodic few-shot learning
    """
    def __init__(self, dataset, n_way, n_shot, n_query):
        self.dataset = dataset
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        
        # Group data by class
        self.class_data = {}
        for idx, (_, label) in enumerate(dataset):
            if label not in self.class_data:
                self.class_data[label] = []
            self.class_data[label].append(idx)
            
        self.classes = list(self.class_data.keys())
        
    def __len__(self):
        # Number of possible episodes
        return 1000  # Arbitrary number of episodes per epoch
        
    def __getitem__(self, idx):
        # Sample n_way classes
        episode_classes = np.random.choice(self.classes, self.n_way, replace=False)
        
        support_samples = []
        query_samples = []
        support_labels = []
        query_labels = []
        
        for i, cls in enumerate(episode_classes):
            # Get indices of samples from this class
            cls_indices = self.class_data[cls]
            
            # Sample n_shot + n_query indices
            selected_indices = np.random.choice(cls_indices, self.n_shot + self.n_query, replace=False)
            
            # Split into support and query
            support_idx = selected_indices[:self.n_shot]
            query_idx = selected_indices[self.n_shot:]
            
            # Get samples
            for idx in support_idx:
                img, _ = self.dataset[idx]
                support_samples.append(img)
                support_labels.append(i)  # Use index as label (0 to n_way-1)
                
            for idx in query_idx:
                img, _ = self.dataset[idx]
                query_samples.append(img)
                query_labels.append(i)  # Use index as label (0 to n_way-1)
                
        # Convert to tensors
        support_samples = torch.stack(support_samples)
        query_samples = torch.stack(query_samples)
        support_labels = torch.tensor(support_labels)
        query_labels = torch.tensor(query_labels)
        
        return support_samples, support_labels, query_samples, query_labels

def train_prototypical(model, train_loader, optimizer, device, epoch):
    """
    Train the prototypical network for one epoch
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (support_samples, support_labels, query_samples, query_labels) in enumerate(train_loader):
        support_samples, support_labels = support_samples.to(device), support_labels.to(device)
        query_samples, query_labels = query_samples.to(device), query_labels.to(device)
        
        n_way = len(torch.unique(support_labels))
        n_shot = support_samples.size(0) // n_way
        
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(support_samples, query_samples, n_way, n_shot)
        
        # Compute loss
        loss = F.cross_entropy(logits, query_labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Compute accuracy
        _, predicted = torch.max(logits.data, 1)
        total += query_labels.size(0)
        correct += (predicted == query_labels).sum().item()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f'Epoch: {epoch} [{batch_idx}/{len(train_loader)}]\tLoss: {loss.item():.6f}\tAccuracy: {100.0 * correct / total:.2f}%')
    
    return total_loss / len(train_loader), 100.0 * correct / total

def test_prototypical(model, test_loader, device):
    """
    Test the prototypical network
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (support_samples, support_labels, query_samples, query_labels) in enumerate(test_loader):
            support_samples, support_labels = support_samples.to(device), support_labels.to(device)
            query_samples, query_labels = query_samples.to(device), query_labels.to(device)
            
            n_way = len(torch.unique(support_labels))
            n_shot = support_samples.size(0) // n_way
            
            # Forward pass
            logits = model(support_samples, query_samples, n_way, n_shot)
            
            # Compute accuracy
            _, predicted = torch.max(logits.data, 1)
            total += query_labels.size(0)
            correct += (predicted == query_labels).sum().item()
    
    return 100.0 * correct / total
