import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def linear_beta_schedule(timesteps, beta_start, beta_end):
    beta_start = beta_start
    beta_end = beta_end
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def exp_beta_schedule(timesteps, beta_min=0.1, beta_max=10):
    x = torch.linspace(1, 2 * timesteps + 1, timesteps)
    betas = 1 - torch.exp(- beta_min / timesteps - x * 0.5 * (beta_max - beta_min) / (timesteps * timesteps))
    return betas

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

class diffusion():
    def __init__(self, timesteps, beta_start, beta_end, beta_type, w, linespace):
        self.timesteps = timesteps
        self.beta_type = beta_type
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.w = w
        self.linespace = linespace

        if self.beta_type == 'linear':
            self.betas = linear_beta_schedule(timesteps=self.timesteps, beta_start=self.beta_start, beta_end=self.beta_end)
        elif self.beta_type == 'exp':
            self.betas = exp_beta_schedule(timesteps=self.timesteps)
        elif self.beta_type =='cosine':
            self.betas = cosine_beta_schedule(timesteps=self.timesteps)
        elif self.beta_type =='sqrt':
            self.betas = torch.tensor(betas_for_alpha_bar(self.timesteps, lambda t: 1-np.sqrt(t + 0.0001),)).float()

        # define alphas 
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        
        # x_t = self.sqrt_alphas_cumprod * x_0 + self.sqrt_one_minus_alphas_cumprod * \epsilon_t
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
        # \epsilon_t = 1.0 / self.sqrt_recipm1_alphas_cumprod * (self.sqrt_recip_alphas_cumprod*x_t - x_0)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)
        
        # DDIM Reverse Process
        indices = list(range(0, self.timesteps+1, self.linespace)) # [0,100,...,2000]
        self.sub_timesteps = len(indices)
        indices_now = [indices[i]-1 for i in range(len(indices))]
        indices_now[0] = 0
        self.alphas_cumprod_ddim = self.alphas_cumprod[indices_now]
        self.alphas_cumprod_ddim_prev = F.pad(self.alphas_cumprod_ddim[:-1], (1, 0), value=1.0)
        self.sqrt_recipm1_alphas_cumprod_ddim = torch.sqrt(1. / self.alphas_cumprod_ddim - 1)

        self.posterior_ddim_coef1 = torch.sqrt(self.alphas_cumprod_ddim_prev) - torch.sqrt(1.-self.alphas_cumprod_ddim_prev)/ self.sqrt_recipm1_alphas_cumprod_ddim
        #self.posterior_ddim_coef1 = (torch.sqrt(self.alphas_cumprod_prev) - (1. - self.alphas_cumprod_prev) / self.sqrt_recipm1_alphas_cumprod)
        self.posterior_ddim_coef2 = torch.sqrt(1.-self.alphas_cumprod_ddim_prev) / torch.sqrt(1. - self.alphas_cumprod_ddim)
        #self.posterior_ddim_coef2 = (1. - self.alphas_cumprod_prev) / torch.sqrt(1. - self.alphas_cumprod)

        # x_{t-1} = self.posterior_mean_coef1 * x_0 + self.posterior_mean_coef2 * x_t + self.posterior_variance * \epsilon
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
    
    def pertube(self, x_start, t, noise=None):
        # print(self.betas)
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def pertube_item(self, x_start, t, noise=None):
        # print(self.betas)
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, denoise_model, target, x_start, h, t, noise=None, loss_type="l2"):
        # 
        if noise is None:
            noise = torch.randn_like(x_start) 
            # noise = torch.randn_like(x_start) / 100
        
        # 
        x_noisy = self.pertube(x_start=x_start, t=t, noise=noise)


        predicted_x = denoise_model(x_noisy, h, t)

        # 
        if loss_type == 'l1':
            loss_diff = F.l1_loss(x_start, predicted_x)
        elif loss_type == 'l2':
            loss_diff = F.mse_loss(x_start, predicted_x)
        elif loss_type == "huber":
            loss_diff = F.smooth_l1_loss(x_start, predicted_x)
        else:
            raise NotImplementedError()
    
        return loss_diff, predicted_x

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    
    @torch.no_grad()
    def p_sample(self, model_forward, model_forward_uncon, x, h, t, t_index):
        # cf guidance
        x_start = (1 + self.w) * model_forward(x, h, t) - self.w * model_forward_uncon(x, t) 
        
        x_t = x 
        model_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)

            return model_mean + torch.sqrt(posterior_variance_t) * noise 
        
    @torch.no_grad()
    def i_sample(self, model_forward, model_forward_uncon, x, h, t, t_index):
        # cf guidance

        x_start = (1 + self.w) * model_forward(x, h, t) - self.w * model_forward_uncon(x, t) 
        
        x_t = x 
        model_mean = (
            self.posterior_ddim_coef1[t_index] * x_start +
            self.posterior_ddim_coef2[t_index] * x_t
        )

        return model_mean 
        
            
    @torch.no_grad()
    def sample_from_noise(self, model_forward, model_forward_uncon, h, hidden_size):

        x = torch.randn(h.shape[0],hidden_size).to(h.device)

        #for n in reversed(range(0, self.timesteps, self.linespace)):
        for n in reversed(range(self.sub_timesteps)):
            step = torch.full((h.shape[0], ), n*self.linespace, device=h.device, dtype=torch.long)
            x = self.i_sample(model_forward, model_forward_uncon, x, h, step, n)

        return x
    
    @torch.no_grad()
    def i_unc_sample(self, model_forward_uncon, x, t, t_index):
        # cf guidance
        x_start = model_forward_uncon(x, t) 
        
        x_t = x 
        model_mean = (
            self.posterior_ddim_coef1[t_index] * x_start +
            self.posterior_ddim_coef2[t_index] * x_t
        )
        return model_mean 
        
            
    @torch.no_grad()
    def unc_sample_from_noise(self, model_forward_uncon, B,device, hidden_size):

        x = torch.randn(B, hidden_size).to(device)

        #for n in reversed(range(0, self.timesteps, self.linespace)):
        for n in reversed(range(self.sub_timesteps)):
            step = torch.full((B, ), n*self.linespace, device=device, dtype=torch.long)
            x = self.i_unc_sample(model_forward_uncon, x, step, n)

        return x
