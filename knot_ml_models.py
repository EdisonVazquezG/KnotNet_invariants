import tensorflow as tf
from tensorflow.keras import layers,models
from tensorflow.keras.optimizers import Adam
import torch
import torch.nn as nn
import torch.optim as optim
import copy

def build_autoencoder(input_size, latent_size, v_n,lr=0.00005):

  ## Encoder
  inputs = layers.Input(shape =(input_size,))
  x = layers.Dense(v_n[0],activation='relu')(inputs)
  x = layers.BatchNormalization()(x)
  x = layers.Dense(v_n[1],activation='relu')(x)
  x = layers.BatchNormalization()(x)
  x = layers.Dense(v_n[2],activation='relu')(x)
  x = layers.BatchNormalization()(x)
  latent = layers.Dense(latent_size, activation='tanh', name='latent_vector')(x)   ## Bottleneck layer

  ## Decoder
  x = layers.Dense(v_n[2],activation='relu')(latent)
  x = layers.BatchNormalization()(x)
  x = layers.Dense(v_n[1],activation='relu')(x)
  x = layers.BatchNormalization()(x)
  x = layers.Dense(v_n[0],activation='relu')(x)
  x = layers.BatchNormalization()(x)
  decoded = layers.Dense(input_size,activation='linear')(x) ##Output layer

  ## Build the autoencoder model
  autoencoder = models.Model(inputs,decoded)
  encoder = models.Model(inputs,latent) ## Encoder part of the autoencoder

  ## Compile the autoencoder
  autoencoder.compile(optimizer=Adam(learning_rate=lr),loss='mse')

  return autoencoder, encoder

#This is for predicting a single direction ---- Example: Alexander->Homfly-PT
class MLP_single(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    def forward(self, x):
        return self.model(x)


#This is for predicting a combined direction ---- Example: Alexander+Jones -> Homfly-PT
class MLP_combined(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
    )
    def forward(self, x):
        return self.model(x)
    
## To train whatever model
def train_model(model, train_loader, test_loader, n_epochs=100, lr=1e-3, patience=5, device='gpu'):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_loss = float('inf')
    best_model = None
    epochs_no_improve = 0

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                output = model(batch_X)
                loss = criterion(output, batch_y)
                test_loss += loss.item()

        train_loss /= len(train_loader)
        test_loss /= len(test_loader)
        print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f}")

        # Early Stopping
        if test_loss < best_loss:
            best_loss = test_loss
            best_model = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"⏹️ Early stopping at epoch {epoch+1}")
                break

    # Load best model
    if best_model:
        model.load_state_dict(best_model)
    return model

def evaluate_model(model, test_loader, criterion, device='cpu'):
    model.eval()
    model.to(device)

    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            output = model(batch_X)
            loss = criterion(output, batch_y)
            total_loss += loss.item()
            all_preds.append(output.cpu())
            all_targets.append(batch_y.cpu())

    final_mse = total_loss / len(test_loader)
    print(f"\n✅ Final Test MSE: {final_mse:.6f}")

    predictions = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    return final_mse, predictions, targets