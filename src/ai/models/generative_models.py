# src/ai/models/generative_models.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# Generative Adversarial Network (GAN)
class Generator(nn.Module):
    def __init__(self, noise_dim, output_dim):
        """Initialize the generator network.

        Args:
            noise_dim (int): Dimension of the noise vector.
            output_dim (int): Dimension of the generated output.
        """
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Tanh()  # Output in range [-1, 1]
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        """Initialize the discriminator network.

        Args:
            input_dim (int): Dimension of the input data.
        """
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output probability
        )

    def forward(self, x):
        return self.model(x)

class GAN:
    def __init__(self, noise_dim, data_dim, learning_rate=0.0002):
        """Initialize the GAN.

        Args:
            noise_dim (int): Dimension of the noise vector.
            data_dim (int): Dimension of the data.
            learning_rate (float): Learning rate for the optimizers.
        """
        self.generator = Generator(noise_dim, data_dim)
        self.discriminator = Discriminator(data_dim)
        self.loss_function = nn.BCELoss()
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=learning_rate)
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=learning_rate)

    def train(self, data_loader, num_epochs, noise_dim):
        """Train the GAN.

        Args:
            data_loader (DataLoader): DataLoader for the training data.
            num_epochs (int): Number of epochs to train.
            noise_dim (int): Dimension of the noise vector.
        """
        for epoch in range(num_epochs):
            for real_data, _ in data_loader:
                batch_size = real_data.size(0)
                real_data = real_data.view(batch_size, -1)  # Flatten the images

                # Train Discriminator
                self.optimizer_D.zero_grad()
                real_labels = torch.ones(batch_size, 1)
                fake_labels = torch.zeros(batch_size, 1)

                output = self.discriminator(real_data)
                d_loss_real = self.loss_function(output, real_labels)

                noise = torch.randn(batch_size, noise_dim)
                fake_data = self.generator(noise)
                output = self.discriminator(fake_data.detach())
                d_loss_fake = self.loss_function(output, fake_labels)

                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                self.optimizer_D.step()

                # Train Generator
                self.optimizer_G.zero_grad()
                output = self.discriminator(fake_data)
                g_loss = self.loss_function(output, real_labels)
                g_loss.backward()
                self.optimizer_G.step()

            print(f"Epoch [{epoch + 1}/{num_epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

    def generate_samples(self, num_samples, noise_dim):
        """Generate samples from the generator.

        Args:
            num_samples (int): Number of samples to generate.
            noise_dim (int): Dimension of the noise vector.

        Returns:
            torch.Tensor: Generated samples.
        """
        noise = torch.randn(num_samples, noise_dim)
        with torch.no_grad():
            generated_samples = self.generator(noise)
        return generated_samples

    def save_models(self, path):
        """Save the generator and discriminator models.

        Args:
            path (str): Path to save the models.
        """
        torch.save(self.generator.state_dict(), os.path.join(path, 'generator.pth'))
        torch.save(self.discriminator.state_dict(), os.path.join(path, 'discriminator.pth'))
        print("Models saved.")

 def load_models(self, path):
        """Load the generator and discriminator models.

        Args:
            path (str): Path to load the models from.
        """
        self.generator.load_state_dict(torch.load(os.path.join(path, 'generator.pth')))
        self.discriminator.load_state_dict(torch.load(os.path.join(path, 'discriminator.pth')))
        print("Models loaded.")

# Example usage
if __name__ == "__main__":
    # Set up data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    noise_dim = 100
    data_dim = 28 * 28  # MNIST images are 28x28
    gan = GAN(noise_dim, data_dim)

    # Train the GAN
    gan.train(data_loader, num_epochs=50, noise_dim=noise_dim)

    # Generate samples
    generated_samples = gan.generate_samples(num_samples=10, noise_dim=noise_dim)
    print("Generated samples shape:", generated_samples.shape)

    # Save models
    gan.save_models('./models')

    # Load models
    gan.load_models('./models')
