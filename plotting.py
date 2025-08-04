"""
    Plotting functions for the diffusion model.

"""
import matplotlib.pyplot as plt
import numpy as np
import torch

from ddm_utils import DiffusionSampler


def show_tensor_image(image):

    # Take first image of batch
    if len(image.shape) == 5:
        image = image[0, :, :, :, :].cpu().numpy()
    # cut off the channel dimension
    image = image[0, :, :, :]
    cut_size = image.shape[0]
    image_2d = image[cut_size//2, :, :]
    plt.imshow(image_2d, cmap='gray')

def show_forward_process(image,forward_fcn,T_m=100, num_images=6):
    """
    Simulates and plots the forward diffusion process for a given image.
    """
    # Args:
    #     image: Image to be diffused
    #     forward_fcn: Forward diffusion sampler with input (image, t)
    #     bae_model: Bezier Autoencoder model
    #     T_m: Number of timesteps to plot
    #     num_images: Number of images to plot
    #     reverse_transform_fcn: Function to reverse scaling/shifting transformation of the latent space.

    # Simulate and plot forward diffusion process            
    plt.figure(figsize=(15,3))
    plt.axis('off')
    num_images = 6
    stepsize = int(T_m//num_images)

    for idx in range(0, T_m, stepsize):
        t = torch.Tensor([idx]).type(torch.int64)
        plt.subplot(2, num_images+1, (idx//stepsize) + 1)
        plt.axis('off')
        # image, noise = forward_fcn(image, t)
        show_tensor_image(forward_fcn(image, t)[0])
        plt.title(f"t={idx}")

    plt.show()


def sample_plot_image(model, epoch, sample_timestep_fcn, dims=(1,1,10,10,10), num_images=6, T=100, animation_mode=False,
                      save=True, save_path='results/',device='cpu', seed=None, figsize=(15,3)):
    """
    Plots a sample image from the diffusion model.
    Args:
        model: Diffusion model
        epoch: Epoch of the model
        sample_timestep_fcn: Function to sample a timestep from the diffusion model
        dims: Dimensions of the image
        num_images: Number of images to plot
        T: Number of timesteps
        animation_mode: If True, saves the image for animation instead of plotting
        noscale: If True, does not scale the image
        save: If True, saves the image
        save_path: Path to save the image
        c: Conditional input to the diffusion model
        reverse_transform: Function to reverse scaling/shifting transformation of the latent space.
        device: Device to run the model on
    """
    with torch.no_grad():
        if seed is not None:
            torch.manual_seed(seed)
        image = torch.randn(dims, device=device) 
        plt.figure(figsize=(15,3))
        plt.axis('off')
        stepsize = int(T/num_images)

        for i in range(0,T)[::-1]:
            t = torch.full((1,), i, device=device, dtype=torch.long)
            image = sample_timestep_fcn(model, image, t)
            if i % stepsize == 0 or i == T-1:
                if not animation_mode:
                    plt.subplot(1, num_images, (i//stepsize)+1)
                plt.axis('off')
                image_plt = image.detach().clone().cpu()
                show_tensor_image(image_plt)
                # save the image of the latent space at step i
                if animation_mode:
                    plt.savefig(save_path + f'latent_space_{i}.png')
                    # clear the figure
                    plt.clf()

        # save image for animation
        if save and not animation_mode:
            plt.savefig(save_path + f'epoch_{epoch}.png')
            plt.show()  

def generate_sample(model, sample_timestep_fcn, dims=(1,1,10,10,10),  T=100, device='cpu', seed=None):
    """
    Generates a sample image from the diffusion model.
    Args:
        model: Diffusion model
        epoch: Epoch of the model
        sample_timestep_fcn: Function to sample a timestep from the diffusion model
        dims: Dimensions of the image
        num_images: Number of images to plot
        T: Number of timesteps
        animation_mode: If True, saves the image for animation instead of plotting
        noscale: If True, does not scale the image
        save: If True, saves the image
        save_path: Path to save the image
        c: Conditional input to the diffusion model
        reverse_transform: Function to reverse scaling/shifting transformation of the latent space.
        device: Device to run the model on
    """
    model.eval()
    with torch.no_grad():
        if seed is not None:
            torch.manual_seed(seed)
        image = torch.randn(dims, device=device) 
        stepsize = int(T/num_images)

        for i in range(0,T)[::-1]:
            t = torch.full((1,), i, device=device, dtype=torch.long)
            image = sample_timestep_fcn(model, image, t)
            
    return image
