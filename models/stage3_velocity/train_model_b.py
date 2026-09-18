import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os

from architectures import ModelB

def train_model_b():
    # Load data
    data_dir = "data/processed/stage3_10hz_50"
    X_train = torch.load(os.path.join(data_dir, "X_train.pt"))
    Y_train = torch.load(os.path.join(data_dir, "Y_train.pt"))
    
    X_test = torch.load(os.path.join(data_dir, "X_test.pt"))
    Y_test = torch.load(os.path.join(data_dir, "Y_test.pt"))
    
    train_dataset = TensorDataset(X_train, Y_train)
    test_dataset = TensorDataset(X_test, Y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    model = ModelB()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 15
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X, Y in train_loader:
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, Y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X.size(0)
            
        train_loss /= len(train_dataset)
        
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for X, Y in test_loader:
                pred = model(X)
                loss = criterion(pred, Y)
                test_loss += loss.item() * X.size(0)
        test_loss /= len(test_dataset)
        
        print(f"Epoch {epoch+1}/{epochs} - Train MSE: {train_loss:.4f}, Test MSE: {test_loss:.4f}")
        
    os.makedirs("models/stage3_velocity/weights", exist_ok=True)
    torch.save(model.state_dict(), "models/stage3_velocity/weights/model_b.pt")
    print("Saved Model B.")

if __name__ == "__main__":
    train_model_b()
