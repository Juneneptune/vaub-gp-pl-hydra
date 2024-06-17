import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid

# Function to display reconstructed images
def display_reconstructed_images(epoch, vae_model, data, n_samples=10, dim=[1, 28, 28], is_flip=False):
    vae_model.eval()
    with torch.no_grad():
        data = data[:n_samples]
        recon_x, z, _, _ = vae_model(data)
        recon_x = recon_x[:n_samples]
        comparison = torch.cat([data.view(-1, dim[0], dim[1], dim[2]), recon_x.view(-1, dim[0], dim[1], dim[2])])
        comparison = make_grid(comparison, nrow=data.size(0))
        comparison = comparison.cpu().numpy().transpose(1, 2, 0)

        plt.figure(figsize=(15, 5))
        plt.imshow(comparison, cmap='gray')
        plt.axis('off')
        plt.title(f'Reconstructed Images at Epoch {epoch}')
        plt.show()


def display_reconstructed_and_flip_images(epoch, vae_model, flip_vae_model, data, n_samples=10, dim=[1, 28, 28],
                                          flip_dim=[3, 32, 32], is_mnist=False, is_both=False):
    vae_model.eval()
    with torch.no_grad():
        data = data[:n_samples]
        recon_x, z, _, _ = vae_model(data)
        recon_x_flip = flip_vae_model.decode(z)
        data = data[:n_samples]
        recon_x = recon_x[:n_samples]
        recon_x_flip = recon_x_flip[:n_samples]

        data = data.view(n_samples, dim[0], dim[1], dim[2])
        recon_x = recon_x.view(n_samples, dim[0], dim[1], dim[2])
        recon_x_flip = recon_x_flip.view(n_samples, flip_dim[0], flip_dim[1], flip_dim[2])
        z = z[:n_samples]
        fig, axes = plt.subplots(3, n_samples, figsize=(n_samples * 3 / 2, 4.5))
        if is_mnist:
            main_color = 'gray'
            flip_color = None
        elif is_both:
            main_color = 'gray'
            flip_color = 'gray'
        else:
            flip_color = 'gray'
            main_color = None

        for i in range(n_samples):
            axes[0, i].imshow(np.transpose(data[i].detach().cpu().numpy(), (1, 2, 0)), cmap=main_color)
            axes[0, i].axis('off')

            axes[1, i].imshow(np.transpose(recon_x[i].detach().cpu().numpy(), (1, 2, 0)), cmap=main_color)
            axes[1, i].axis('off')

            axes[2, i].imshow(np.transpose(recon_x_flip[i].detach().cpu().numpy(), (1, 2, 0)), cmap=flip_color)
            axes[2, i].axis('off')

    return plt


def extract_top_k_features(data, k, is_pca=False, is_both=False):
    """
    Perform PCA on the input data and extract the most important features based on the top k principal components.
    
    Args:
        data (torch.Tensor): Input data matrix of shape (n_samples, n_features)
        k (int): Number of top principal components to consider

    Returns:
        torch.Tensor: The data reduced to the most important features
        torch.Tensor: The indices of the most important features
    """
    # Center the data by subtracting the mean of each feature
    data_mean = torch.mean(data, dim=0)
    data_centered = data - data_mean

    # Perform SVD on the centered data
    U, S, V = torch.svd(data_centered)

    # The top k principal components are the first k columns of V
    top_k_components = V[:, :k]

    # Compute the importance of each feature by the magnitude of the loadings
    feature_top_k = torch.sum(top_k_components**2, dim=1)

    # Get the indices of the most important features
    top_k_features_indices = torch.argsort(feature_top_k, descending=True)

    # Select the data indexed by the important features
    top_k_data = data[:, top_k_features_indices[:k]]
    top_k_pca = torch.matmul(data_centered, top_k_components)

    if is_pca:
        return top_k_pca
    elif is_both:
        return top_k_data, top_k_pca
    else:
        return top_k_data

def plt_scatter_alignment(X, k=2, is_pca=False, is_both=False):
    if is_both:
        x_reduced, x_reduced_pca = extract_top_k_features(X, k, is_both=True)
        x1_reduced, x2_reduced = x_reduced.chunk(2)
        x1_reduced_pca, x2_reduced_pca = x_reduced_pca.chunk(2)
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].set_title('Top K PCA')
        axs[1].set_title('Top K Features')
    elif is_pca:
        x1_reduced, x2_reduced = extract_top_k_features(X, k, is_pca=True).chunk(2)
        plt.set_title('Top K PCA')
    else:
        x1_reduced, x2_reduced = extract_top_k_features(X, k).chunk(2)
        plt.set_title('Top K Features')

    if is_both:
        x1_reduced, x2_reduced = x1_reduced.detach().cpu().numpy(), x2_reduced.detach().cpu().numpy()
        x1_reduced_pca, x2_reduced_pca = x1_reduced_pca.detach().cpu().numpy(), x2_reduced_pca.detach().cpu().numpy()
        axs[0].scatter(x1_reduced[:, 0], x1_reduced[:, 1], c='r', marker='x')
        axs[0].scatter(x2_reduced[:, 0], x2_reduced[:, 1], c='b', marker='+')
        axs[1].scatter(x1_reduced_pca[:, 0], x1_reduced_pca[:, 1], c='r', marker='x')
        axs[1].scatter(x2_reduced_pca[:, 0], x2_reduced_pca[:, 1], c='b', marker='+')
    else:
        x1_reduced, x2_reduced = x1_reduced.detach().cpu().numpy(), x2_reduced.detach().cpu().numpy()
        plt.scatter(x1_reduced[:, 0], x1_reduced[:, 1], c='r', marker='x')
        plt.scatter(x2_reduced[:, 0], x2_reduced[:, 1], c='b', marker='+')
        
    return plt