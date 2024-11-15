# Diffusion added from https://github.com/XiangLi1999/Diffusion-LM/blob/main/improved-diffusion/improved_diffusion/gaussian_diffusion.py
import enum
import math
from typing import Dict, Optional
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import HeteroData
from smart.modules.agent_decoder import SMARTAgentDecoder
from smart.modules.diff_agent_decoder import DiffSMARTAgentDecoder
from smart.modules.discrete_diffusion import UnifiedDiscreteDiffusion
from smart.modules.map_decoder import SMARTMapDecoder
from diffusers import DDPMScheduler
### UTILS DIFFUSION ###
def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    elif schedule_name == 'sqrt':
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: 1-np.sqrt(t + 0.0001),
        )
    elif schedule_name == "trunc_cos":
        return betas_for_alpha_bar2(
            num_diffusion_timesteps,
            lambda t: np.cos((t + 0.1) / 1.1 * np.pi / 2) ** 2,
        )
    elif schedule_name == 'trunc_lin':
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001 + 0.01
        beta_end = scale * 0.02 + 0.01
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == 'pw_lin':
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001 + 0.01
        beta_mid = scale * 0.0001  #scale * 0.02
        beta_end = scale * 0.02
        first_part = np.linspace(
            beta_start, beta_mid, 10, dtype=np.float64
        )
        second_part = np.linspace(
            beta_mid, beta_end, num_diffusion_timesteps - 10 , dtype=np.float64
        )
        return np.concatenate(
            [first_part, second_part]
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")
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
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))
class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB
    E2E_KL = enum.auto()
    E2E_MSE = enum.auto()
    E2E_Simple_MSE = enum.auto()
    E2E_Simple_KL = enum.auto()

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


### GUASSIAN DIFFUSION ###
class SMARTDiffusion(nn.Module):
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/XiangLi1999/Diffusion-LM/blob/main/improved-diffusion/improved_diffusion/gaussian_diffusion.py

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        dataset: str,
        input_dim: int,
        hidden_dim: int,
        num_historical_steps: int,
        pl2pl_radius: float,
        time_span: Optional[int],
        pl2a_radius: float,
        a2a_radius: float,
        num_freq_bands: int,
        num_map_layers: int,
        num_agent_layers: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
        map_token: Dict,
        token_data: Dict,
        use_intention=False,
        token_size=512,
        num_diffusion_timesteps=10,
        schedule_name="cosine",
        betas=None,
        model_mean_type=ModelMeanType.START_X,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.E2E_MSE,
        
    ):
        super(SMARTDiffusion, self).__init__()

        #region Diffusion Constants
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.num_diffusion_timesteps = num_diffusion_timesteps

        #region SMART Diffusion Based Decoder
        self.map_encoder = SMARTMapDecoder(
            dataset=dataset,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            pl2pl_radius=pl2pl_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_map_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            map_token=map_token
        )
        self.agent_encoder = DiffSMARTAgentDecoder(
                dataset=dataset,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_historical_steps=num_historical_steps,
                time_span=time_span,
                pl2a_radius=pl2a_radius,
                a2a_radius=a2a_radius,
                num_freq_bands=num_freq_bands,
                num_layers=num_agent_layers,
                num_heads=num_heads,
                head_dim=head_dim,
                dropout=dropout,
                token_size=token_size,
                token_data=token_data
            )
        
        self.vocab_embedding = torch.nn.Embedding(token_size, hidden_dim)
        self.embedding_normalizer = torch.nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.linear = nn.Linear(hidden_dim, token_size, bias=True)

        self.diffusion = UnifiedDiscreteDiffusion(self.num_diffusion_timesteps, # 0 means use continuous time
                                     num_classes=token_size, 
                                     noise_schedule_type=schedule_name,
                                     noise_schedule_args={},
                                     )
        #endregion
    ### TRAINING ###
    def q_sample(self, x_start, t, noise):        
        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        return sample    

    def p_losses(self, x_start_init, t):
        noise_init = torch.randn_like(x_start_init)
        x_start = x_start_init
        noise = noise_init
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        batch_size = data['agent']['batch'].max().item() + 1
        device = data['agent']['batch'].device
        agent_batch_association = data['agent']['batch']
        # Map encodings
        map_enc = self.map_encoder(data)


        timesteps = torch.randint(1, self.num_diffusion_timesteps, (batch_size,), device=device).long()
        # add noise 
        timesteps_per_agent = timesteps[agent_batch_association]
        x_0 = data['agent']['token_idx']
        x_t = self.diffusion.qt_0_sample(x_0, timesteps_per_agent)
        agent_enc = self.agent_encoder(data, map_enc, x_noisy=x_t, timesteps=timesteps)        
        return {
            **map_enc, 
            **agent_enc,
            'logits_t': agent_enc['token_prob'],
            'x_t': x_t,
            'x_0': x_0,
            't': timesteps_per_agent,
            'conditional_mask': agent_enc['token_eval_mask']
            }

    def inference(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        map_enc = self.map_encoder(data)

        # model_input 
        agent_token_index = data['agent']['token_idx'].clone()
        agent_token_mask_conditioned = torch.full_like(agent_token_index, True, dtype=torch.bool)
        agent_token_mask_conditioned[:, 2:] = False # first 2 tokens are kept
        agent_token_index[~agent_token_mask_conditioned] = 0 # I am removing all the tokens in the future. as the model has to predict them 
        m_noise_distribution = torch.full((*agent_token_index.shape , self.agent_encoder.token_size), 1.0, device=agent_token_index.device, dtype=torch.float)

        class DenoisingModule(nn.Module):
            def __init__(self, agent_encoder):
                super(DenoisingModule, self).__init__()
                self.agent_encoder = agent_encoder

            def forward(self, x_t, timesteps):
                output = self.agent_encoder(data, map_enc, x_noisy=x_t, timesteps=timesteps)
                logits = output['token_prob']
                return logits
        # wraper for the denoising function
        denoiser = DenoisingModule(self.agent_encoder)

        x_t = self.diffusion.sample(denoiser,
            num_backward_steps=self.num_diffusion_timesteps,
            m = m_noise_distribution,
            conditional_mask=agent_token_mask_conditioned,
            conditional_input=agent_token_index
        )

        info_dict = self.agent_encoder.inference(data, map_enc, generated_token_idx=x_t)
        # Go from tokens entire path tokens to the actull information
        
        return {**map_enc, **info_dict, 'x_t': x_t}
        

class SMARTDecoder(nn.Module):

    def __init__(self,
                 dataset: str,
                 input_dim: int,
                 hidden_dim: int,
                 num_historical_steps: int,
                 pl2pl_radius: float,
                 time_span: Optional[int],
                 pl2a_radius: float,
                 a2a_radius: float,
                 num_freq_bands: int,
                 num_map_layers: int,
                 num_agent_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float,
                 map_token: Dict,
                 token_data: Dict,
                 use_intention=False,
                 token_size=512) -> None:
        super(SMARTDecoder, self).__init__()
        self.diffusion = True
        self.map_encoder = SMARTMapDecoder(
            dataset=dataset,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            pl2pl_radius=pl2pl_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_map_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            map_token=map_token
        )

        self.agent_encoder = SMARTAgentDecoder(
            dataset=dataset,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            time_span=time_span,
            pl2a_radius=pl2a_radius,
            a2a_radius=a2a_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_agent_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            token_size=token_size,
            token_data=token_data
        )
        self.map_enc = None

    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        map_enc = self.map_encoder(data)
        agent_enc = self.agent_encoder(data, map_enc)
        return {**map_enc, **agent_enc}

    def inference(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        map_enc = self.map_encoder(data)
        agent_enc = self.agent_encoder.inference(data, map_enc)
        return {**map_enc, **agent_enc}

    def inference_no_map(self, data: HeteroData, map_enc) -> Dict[str, torch.Tensor]:
        agent_enc = self.agent_encoder.inference(data, map_enc)
        return {**map_enc, **agent_enc}
