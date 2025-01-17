"""" 
    File for the diffusion models utilities, such as the linear beta schedule and the forward diffusion sample.
    Contains:
        - beta_schedule
        

"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm

def beta_schedule(T, start=1e-4, end=0.02, scale= 1.0, cosine=False, exp_biasing=False, exp_bias_factor=1):
    """
    Returns a beta schedule (default: linear) for the diffusion model.
    Args:
        T: Number of timesteps
        start: Starting value of beta
        end: Ending value of beta
        scale: Scaling factor for beta
        cosine: Whether to use a cosine beta schedule
        exp_biasing: Whether to use exponential biasing
        exp_bias_factor: Exponential biasing factor
    """

    beta = torch.linspace(scale*start, scale*end, T)
    if cosine:
        beta = []
        a_func = lambda t_val: math.cos((t_val + 0.008) / 1.008 * np.pi / 2) ** 2
        for i in range(T):
            t1 = i / T
            t2 = (i + 1) / T
            beta.append(min(1 - a_func(t2) / a_func(t1), 0.999))
        
        beta = torch.tensor(beta)
    
    if exp_biasing:
        beta = (torch.flip(torch.exp(-exp_bias_factor*torch.linspace(0, 1, T)), dims=[0]))*beta

    return beta

def get_index_from_list(vals, t, x_shape):
    """ 
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    Credit: 
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

class DiffusionSampler():
    # Precompute the sqrt alphas and sqrt one minus alphas
    def __init__(self, T, betas):
        self.T = T
        self.betas = betas
        self.alphas = (1. - self.betas)
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def forward_diffusion_sample(self, x_0, t, device="cpu"):
        """ 
        Takes an image and a timestep as input and 
        returns the noisy version of it
        """
        noise = torch.randn_like(x_0).to(device)
        sqrt_alphas_cumprod_t = get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )

        # mean + variance
        return sqrt_alphas_cumprod_t.to(device) * x_0.to(device)\
        + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)
    
    def forward_diffusion_sample_partial(self, x_0, t_current, t_final, device="cpu"):
        """ 
        Takes an image at a timestep and
        adds noise to reach the desired timestep
        """
        for i in range(t_final[0]-t_current[0]):
            t = t_final - i

            noise = torch.randn_like(x_0, ).to(device)
            x_0 = torch.sqrt(get_index_from_list(self.alphas, t, x_0.shape)) * x_0.to(device)\
            + torch.sqrt(get_index_from_list(1-self.alphas, t, x_0.shape)) * noise.to(device)

        # sqrt_alphas_cumprod_t_current = get_index_from_list(self.sqrt_alphas_cumprod, t_current, x_0.shape)
        # sqrt_one_minus_alphas_cumprod_t_current = get_index_from_list(
        #     self.sqrt_one_minus_alphas_cumprod, t_current, x_0.shape
        # )
        # sqrt_alphas_cumprod_t_final = get_index_from_list(self.sqrt_alphas_cumprod, t_final, x_0.shape)
        # sqrt_one_minus_alphas_cumprod_t_final = get_index_from_list(
        #     self.sqrt_one_minus_alphas_cumprod, t_final, x_0.shape
        # )
        # mean + variance
        return x_0, noise.to(device)
    
    def plot_sqrt_alpha_vs_t(self, x_0, t, device="cpu"):
        """ 
        plots the sqrt alphas and sqrt one minus alphas, 
        the amount of noise for each timestep
        """
        cumprods = []
        cumprods_m_one = []
        cumprods_sum = []
        for i in range(t):
            cumprods.append(float(get_index_from_list(self.sqrt_alphas_cumprod, torch.tensor([i]), x_0.shape)))
            cumprods_m_one.append(float(get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, torch.tensor([i]), x_0.shape
            )))
            cumprods_sum.append(cumprods[-1]+cumprods_m_one[-1])
        plt.plot(cumprods, label="sqrt_alphas_cumprod")
        plt.plot(cumprods_m_one, label="sqrt_one_minus_alphas_cumprod")
        plt.plot(cumprods_sum, label="sum")
        plt.legend()
        plt.show()
        # noise = torch.randn_like(x_0).to(device)
        # sqrt_alphas_cumprod_t = get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        # sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        #     self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        # )

        # mean + variance
        # return sqrt_alphas_cumprod_t.to(device) * x_0.to(device)\
        # + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)

    def diffusion_step_sample(self, noise_pred, x_noisy,  t, device="cpu"):
        """ 
        Takes an image, noise and step; returns denoised image. 
        """
        betas_t = get_index_from_list(self.betas, t, x_noisy.shape).to(device)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x_noisy.shape
        ).to(device)
        sqrt_recip_alphas_t = get_index_from_list(self.sqrt_recip_alphas, t, x_noisy.shape).to(device)
        model_mean = sqrt_recip_alphas_t * (
            x_noisy - betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t
        )
        posterior_variance_t = get_index_from_list(self.posterior_variance, t, x_noisy.shape).to(device)

        # mean + variance
        return (model_mean + torch.sqrt(posterior_variance_t) * noise_pred).to(device)
    
    def lossfn_builder(self):
        """
        Returns the loss function for the diffusion model.
        """
        def lossfn(noise_pred, noise):

            return F.mse_loss(noise_pred, noise)
        
        return lossfn

    def sample_timestep(self, model, x, t, c=None, t_mask=None):
        """
        Calls the model to predict the noise in the image and returns 
        the denoised image. 
        Applies noise to this image, if we are not in the last step yet.
        """
        model.eval()
        with torch.no_grad():
            betas_t = get_index_from_list(self.betas, t, x.shape)
            sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
                self.sqrt_one_minus_alphas_cumprod, t, x.shape
            )
            sqrt_recip_alphas_t = get_index_from_list(self.sqrt_recip_alphas, t, x.shape)
            
            # Call model (current image - noise prediction)
            if c is not None:
                # with torch.cuda.amp.autocast(dtype=torch.float16):
                model_mean = sqrt_recip_alphas_t * (
                    x - betas_t * model(x, t, c) / sqrt_one_minus_alphas_cumprod_t
                )
            else:
                current_test_batch_size = x.shape[0]
                dev = x.device
                encoder_hidden_states=torch.ones((current_test_batch_size,1,32)).to(dev)
              
                # with torch.cuda.amp.autocast(dtype=torch.float16):
                model_mean = sqrt_recip_alphas_t * (
                    x - betas_t * model(x, t, encoder_hidden_states).sample / sqrt_one_minus_alphas_cumprod_t
                )

            posterior_variance_t = get_index_from_list(self.posterior_variance, t, x.shape)
            if t_mask is None:
                device = x.device
               
                t_mask = ((t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))).to(device) 
            
            return model_mean + torch.sqrt(posterior_variance_t) * torch.randn_like(x) * t_mask


            # if t == 0:
            #     return model_mean
            # else:
            #     noise = torch.randn_like(x)
            #     return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    def sample_timestep_masked(self, model, x, gt, t, mask):
        # Implements the RePaint sampling procedure to produce an inpainted image based on a known part of the image and a mask.
        # The mask is a binary tensor of the same size as the image, with 1s where the image is known and 0s where it is unknown.
        # gt = ground truth at x0 with arbitrary values in the unknown part

        model.eval()
        with torch.no_grad():
            alpha_cumprod_t = get_index_from_list(self.alphas_cumprod, t, x.shape)
            sqrt_alpha_cumprod_t = get_index_from_list(self.sqrt_alphas_cumprod, t, x.shape)
            
            gt_weight = sqrt_alpha_cumprod_t
            gt_part = gt_weight * gt

            noise_weight = get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
            # ts in shape of noise_weight if t=0 then noise_weight = 0
            dev = x.device

            t_mask = ((t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))).to(dev) 
            noise_part = noise_weight * torch.randn_like(x) * t_mask

            weighted_gt = gt_part + noise_part

            x = mask * weighted_gt + (1 - mask) * x

            x_final = self.sample_timestep(model, x, t, t_mask=t_mask)
        
        return x_final
    
    def generate_sample(self, model, gt, T, mask,seed=None):
        """
        Generates a sample image from the diffusion model.
        Args:
            model: Diffusion model
            gt: Initial ground truth image with noise in the unknown part (can be arbitrary values too)
            T: tuple, (start timesetp, end timestep)
            mask: Binary tensor of the same size as the image, with 1s where the image is known and 0s where it is unknown.
        """
        device = gt.device
        model.eval()
        with torch.no_grad():
            if seed is not None:
                torch.manual_seed(seed)
            
            dims = gt.shape
            # print(dims)
            image = torch.randn(dims, device=device) # initial image
            x_batch = image.shape[0]

            for i in range(T)[::-1]:
                t = torch.full((x_batch,), i, device=device, dtype=torch.long)
                if i == T-1:
                    # print("timestep", i)
                    image = self.sample_timestep(model, image, t)
                else:
                    image = self.sample_timestep_masked(model, image, gt, t, mask)
                
        return image
    
    def generate_sample_resample(self, model, gt, T, mask, N=10, jump=1, seed=None):
        """
        Generates a sample image from the diffusion model.
        Args:
            model: Diffusion model
            gt: Initial ground truth image with noise in the unknown part (can be arbitrary values too)
            T: tuple, (start timesetp, end timestep)
            mask: Binary tensor of the same size as the image, with 1s where the image is known and 0s where it is unknown.
            N: number of resampling steps
            jump: jump size
        """
        device = gt.device
        model.eval()
        with torch.no_grad():
            if seed is not None:
                torch.manual_seed(seed)
            
            dims = gt.shape
            image = torch.randn(dims, device=device) # initial image
            x_batch = image.shape[0]

            # # print(999)
            # t = torch.full((x_batch,), 999, device=device, dtype=torch.long)
            # image = self.sample_timestep(model, image, t)
            # for i in range(jump, T-1-100, jump):
            #     # print(999-i)
            #     t = torch.full((x_batch,), 999-i, device=device, dtype=torch.long)

            #     image = self.sample_timestep_masked(model, image, gt, t, mask)

            #     for j in range(N):
            #         # t = torch.full((x_batch,), i, device=device, dtype=torch.long)
            #         # print("backward", i-j)
                    
            #         # print(999-i-jump)
            #         image = self.sample_timestep_masked(model, image, gt, t-jump, mask)
            #         # print(999-i)
            #         image = self.forward_diffusion_sample_partial(image, t-jump, t, device)[0]
                   
 
            for i in range(25, T-1, jump)[::-1]:
                # print(i)
                for k in range(jump)[::-1]:
                    t = torch.full((x_batch,), i+k, device=device, dtype=torch.long)
                    if i == T-1:
                        # print("timestep", i)
                        image = self.sample_timestep(model, image, t)
                    else:
                        image = self.sample_timestep_masked(model, image, gt, t, mask)

                for j in range(N):
                    t = torch.full((x_batch,), i, device=device, dtype=torch.long)
                    # print("backward", i-j)
                    if i == T-1:
                        # print("timestep", i)
                        image = self.sample_timestep(model, image, t)
                    else:
                        image = self.sample_timestep_masked(model, image, gt, t, mask)
                    image = self.forward_diffusion_sample_partial(image, t, t+1, device)[0]
                # np.save("run_107_batch_1_step_{}.npy".format(i), image.cpu().numpy())

                

            for i in range(0, 25)[::-1]:
                # print(i)
                t = torch.full((x_batch,), i, device=device, dtype=torch.long)
                if i == T-1:
                    # print("timestep", i)
                    image = self.sample_timestep(model, image, t)
                else:
                    image = self.sample_timestep_masked(model, image, gt, t, mask)

                    
        return image

    def generate_partial(self, model, gt, T, mask,seed=None):
        """
        Generates a sample image from the diffusion model.
        Args:
            model: Diffusion model
            gt: Initial ground truth image with noise in the unknown part (can be arbitrary values too)
            T: tuple, (start timesetp, end timestep)
            mask: Binary tensor of the same size as the image, with 1s where the image is known and 0s where it is unknown.
        """
        device = gt.device
        model.eval()
        with torch.no_grad():
            if seed is not None:
                torch.manual_seed(seed)
            
            dims = gt.shape
            # print(dims)
            image = gt
            x_batch = image.shape[0]

            for i in range(T[0],T[1])[::-1]:
                t = torch.full((x_batch,), i, device=device, dtype=torch.long)
                if i == T[1]-1:
                    # print("timestep", i)
                    image = self.sample_timestep(model, image, t)
                else:
                    image = self.sample_timestep_masked(model, image, gt, t, mask)
                
        return image
    
    def animate_sample(self, model, gt, T, mask,seed=None, vrange = [-1.5,1.5], cmap="gray", save_path=None, cutx=0, skip=25):
        """
        Generates a sample image from the diffusion model.
        Args:
            model: Diffusion model
            gt: Initial ground truth image with noise in the unknown part (can be arbitrary values too)
            T: Number of timesteps
            mask: Binary tensor of the same size as the image, with 1s where the image is known and 0s where it is unknown.
        """
        device = gt.device
        model.eval()
        with torch.no_grad():
            if seed is not None:
                torch.manual_seed(seed)
            
            dims = gt.shape
            # print(dims)
            image = torch.randn(dims, device=device) # initial image
            x_batch = image.shape[0]

            for i in range(0,T)[::-1]:
                t = torch.full((x_batch,), i, device=device, dtype=torch.long)
                if i == T-1:
                    # print("timestep", i)
                    image = self.sample_timestep(model, image, t)
                else:
                    image = self.sample_timestep_masked(model, image, gt, t, mask)
                
                if i % skip == 0 or i == T-1:
                    image_plt = image[0].cpu().numpy()
                    plt.axis('off')
                    if vrange is not None:
                        plt.imshow(image_plt[0,cutx,:,:], cmap=cmap, vmin=vrange[0], vmax=vrange[1])
                    else:
                        plt.imshow(image_plt[0,cutx,:,:], cmap=cmap)
                    plt.savefig(save_path + str(i) + ".png")
                
        return image

    def animate_sample_3d(self, model, gt, T, mask,seed=None, vrange = [-1.5,1.5], cmap="gray", save_path=None, cutx=0, skip=25):
        """
        Generates a sample image from the diffusion model.
        Args:
            model: Diffusion model
            gt: Initial ground truth image with noise in the unknown part (can be arbitrary values too)
            T: Number of timesteps
            mask: Binary tensor of the same size as the image, with 1s where the image is known and 0s where it is unknown.
        """
        device = gt.device
        model.eval()
        with torch.no_grad():
            if seed is not None:
                torch.manual_seed(seed)
            
            dims = gt.shape
            # print(dims)
            image = torch.randn(dims, device=device) # initial image
            x_batch = image.shape[0]

            for i in range(0,T)[::-1]:
                t = torch.full((x_batch,), i, device=device, dtype=torch.long)
                if i == T-1:
                    # print("timestep", i)
                    image = self.sample_timestep(model, image, t)
                else:
                    image = self.sample_timestep_masked(model, image, gt, t, mask)
                
                if i % skip == 0 or i == T-1:
                    image_plt = image[0].cpu().numpy()
                    plt.axis('off')
                    ax = plt.figure().add_subplot(projection='3d')
                    facecolors = cm.viridis(image_plt[0,:,:])
                    ax.voxels(image_plt[0,:,:], facecolors=facecolors)
                    plt.axis('off')
                    plt.savefig(save_path + str(i) + "_3d.png")
                
        return image