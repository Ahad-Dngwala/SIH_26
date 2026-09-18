import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os

from architectures import ModelD

def nll_loss(mu, var, target):
    # NLL of Gaussian: 0.5 * [ log(var) + (target - mu)^2 / var ] + C
    loss = 0.5 * (torch.log(var) + (target - mu)**2 / var)
    return loss.mean()

def train_model_d():
    data_dir = "data/processed/stage3_10hz_50"
    X_train = torch.load(os.path.join(data_dir, "X_train.pt"))
    Y_train = torch.load(os.path.join(data_dir, "Y_train.pt"))
    
    X_test = torch.load(os.path.join(data_dir, "X_test.pt"))
    Y_test = torch.load(os.path.join(data_dir, "Y_test.pt"))
    
    ukf_speed_train = Y_train + torch.randn_like(Y_train) * 2.0
    ukf_speed_test = Y_test + torch.randn_like(Y_test) * 2.0
    
    delta_v_train = Y_train - ukf_speed_train
    delta_v_test = Y_test - ukf_speed_test
    
    train_dataset = TensorDataset(X_train, ukf_speed_train, delta_v_train)
    test_dataset = TensorDataset(X_test, ukf_speed_test, delta_v_test)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    model = ModelD()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 15
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X, ukf_v, target_dv in train_loader:
            optimizer.zero_grad()
            mu, var = model(X, ukf_v)
            loss = nll_loss(mu, var, target_dv)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X.size(0)
            
        train_loss /= len(train_dataset)
        
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for X, ukf_v, target_dv in test_loader:
                mu, var = model(X, ukf_v)
                loss = nll_loss(mu, var, target_dv)
                test_loss += loss.item() * X.size(0)
        test_loss /= len(test_dataset)
        
        print(f"Epoch {epoch+1}/{epochs} - Train NLL: {train_loss:.4f}, Test NLL: {test_loss:.4f}")
        
    os.makedirs("models/stage3_velocity/weights", exist_ok=True)
    torch.save(model.state_dict(), "models/stage3_velocity/weights/model_d.pt")
    print("Saved Model D.")

if __name__ == "__main__":
    train_model_d()
