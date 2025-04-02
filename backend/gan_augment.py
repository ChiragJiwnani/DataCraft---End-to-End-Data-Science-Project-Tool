#gan_augment.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define the Generator Model
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Define the Discriminator Model
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Function to Train GAN and Generate Synthetic Data
def train_gan(real_data, epochs=5000, batch_size=64, noise_dim=10):
    num_features = real_data.shape[1]

    generator = Generator(noise_dim, num_features)
    discriminator = Discriminator(num_features)

    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=0.001)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.001)

    for epoch in range(epochs):
        real_samples = torch.tensor(real_data.sample(batch_size, replace=True).values, dtype=torch.float32)
        noise = torch.randn(batch_size, noise_dim)
        fake_samples = generator(noise)

        # Train Discriminator
        optimizer_d.zero_grad()
        real_preds = discriminator(real_samples)
        fake_preds = discriminator(fake_samples.detach())
        loss_d = criterion(real_preds, torch.ones_like(real_preds)) + criterion(fake_preds, torch.zeros_like(fake_preds))
        loss_d.backward()
        optimizer_d.step()

        # Train Generator
        optimizer_g.zero_grad()
        fake_preds = discriminator(fake_samples)
        loss_g = criterion(fake_preds, torch.ones_like(fake_preds))
        loss_g.backward()
        optimizer_g.step()

        if epoch % 1000 == 0:
            logging.info(f"Epoch {epoch}/{epochs} | Loss D: {loss_d.item():.4f} | Loss G: {loss_g.item():.4f}")

    # Generate Final Synthetic Data
    noise = torch.randn(len(real_data), noise_dim)
    synthetic_data = generator(noise).detach().numpy()
    
    # Convert to DataFrame
    synthetic_df = pd.DataFrame(synthetic_data, columns=real_data.columns)
    
    return synthetic_df

# Flask Route to Generate Augmented Data
def generate_synthetic_data(data_path):
    df = pd.read_csv(data_path)
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        return None, "No numeric data found for augmentation"
    
    synthetic_df = train_gan(numeric_df)
    
    augmented_filepath = os.path.join(os.path.dirname(data_path), 'augmented_data.csv')
    synthetic_df.to_csv(augmented_filepath, index=False)
    
    return augmented_filepath, synthetic_df
