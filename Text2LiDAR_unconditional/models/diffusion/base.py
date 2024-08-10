from typing import List, Literal
import einops
import torch
import pywt
from torch import nn
from torch.cuda.amp import autocast
import matplotlib.pyplot as plt

def get_multisize(target):
    target_2 = torch.nn.functional.interpolate(target, scale_factor=1/2, mode='nearest')
    target_4 = torch.nn.functional.interpolate(target, scale_factor=1/4, mode='nearest')
    target_8 = torch.nn.functional.interpolate(target, scale_factor=1/8, mode='nearest')
    target_16 = torch.nn.functional.interpolate(target, scale_factor=1/16, mode='nearest')
    return [target_2, target_4, target_8, target_16]

def get_multimask(loss_mask):
    loss_mask_2 = torch.nn.functional.interpolate(loss_mask, scale_factor=1/2, mode='nearest')
    loss_mask_4 = torch.nn.functional.interpolate(loss_mask, scale_factor=1/4, mode='nearest')
    loss_mask_8 = torch.nn.functional.interpolate(loss_mask, scale_factor=1/8, mode='nearest')
    loss_mask_16 = torch.nn.functional.interpolate(loss_mask, scale_factor=1/16, mode='nearest')
    return [loss_mask_2, loss_mask_4, loss_mask_8, loss_mask_16]

class GaussianDiffusion(nn.Module):
    """
    Base class for continuous/discrete Gaussian diffusion models
    """

    def __init__(
        self,
        denoiser: nn.Module,
        sampling: Literal["ddpm", "ddim"] = "ddpm",
        criterion: Literal["l2", "l1", "huber"] | nn.Module = "l2",
        num_training_steps: int = 1000,
        objective: Literal["eps", "v", "x0"] = "eps",
        beta_schedule: Literal["linear", "cosine", "sigmoid"] = "linear",
        min_snr_loss_weight: bool = True,
        min_snr_gamma: float = 5.0,
        sampling_resolution: tuple[int, int] | None = None,
        clip_sample: bool = True,
        clip_sample_range: float = 1,
    ):
        super().__init__()
        self.denoiser = denoiser
        self.sampling = sampling
        self.num_training_steps = num_training_steps
        self.objective = objective
        self.beta_schedule = beta_schedule
        self.min_snr_loss_weight = min_snr_loss_weight
        self.min_snr_gamma = min_snr_gamma
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range

        if criterion == "l2":
            self.criterion = nn.MSELoss(reduction="none")
        elif criterion == "l1":
            self.criterion = nn.L1Loss(reduction="none")
        elif criterion == "huber":
            self.criterion = nn.SmoothL1Loss(reduction="none")
        elif isinstance(criterion, nn.Module):
            self.criterion = criterion
        else:
            raise ValueError(f"invalid criterion: {criterion}")
        if hasattr(self.criterion, "reduction"):
            assert self.criterion.reduction == "none"

        if sampling_resolution is None:
            assert hasattr(self.denoiser, "resolution")
            assert hasattr(self.denoiser, "in_channels")
            self.sampling_shape = (self.denoiser.in_channels, *self.denoiser.resolution)
        else:
            assert len(sampling_resolution) == 2
            assert hasattr(self.denoiser, "in_channels")
            self.sampling_shape = (self.denoiser.in_channels, *sampling_resolution)

        self.setup_parameters()
        self.register_buffer("_dummy", torch.tensor([]))

    @property
    def device(self):
        return self._dummy.device

    def randn(
        self,
        *shape,
        rng: List[torch.Generator] | torch.Generator | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if rng is None:
            return torch.randn(*shape, **kwargs)
        elif isinstance(rng, torch.Generator):
            return torch.randn(*shape, generator=rng, **kwargs)
        elif isinstance(rng, list):
            assert len(rng) == shape[0]
            return torch.stack(
                [torch.randn(*shape[1:], generator=r, **kwargs) for r in rng]
            )
        else:
            raise ValueError(f"invalid rng: {rng}")

    def randn_like(
        self,
        x: torch.Tensor,
        rng: List[torch.Generator] | torch.Generator | None = None,
    ) -> torch.Tensor:
        return self.randn(*x.shape, rng=rng, device=x.device, dtype=x.dtype)

    def setup_parameters(self) -> None:
        raise NotImplementedError

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        raise NotImplementedError

    @torch.inference_mode()
    def p_sample(self, *args, **kwargs):
        raise NotImplementedError

    @autocast(enabled=False)
    def q_sample(self, x_0, steps, noise):
        raise NotImplementedError

    def get_denoiser_condition(self, steps: torch.Tensor):
        raise NotImplementedError

    def get_target(self, x_0, steps, noise):
        raise NotImplementedError

    def get_loss_weight(self, steps):
        raise NotImplementedError

    def p_loss(
        self,
        x_0: torch.Tensor,
        steps: torch.Tensor,
        loss_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # shared in continuous/discrete versions
        # loss_weights = [1, 0.1, 0.1, 0.05, 0.0] # 5decoder
        # plt.imshow(x_0[0,0].cpu().detach().numpy(), cmap="Greys")
        # plt.savefig('./debug-fig/x_0.png', dpi=300, bbox_inches='tight', pad_inches=0)
        device = x_0.device
        loss_weights = [1, 0.0, 0.0, 0.0, 0.0, 0.1]
        loss_mask = torch.ones_like(x_0) if loss_mask is None else loss_mask
        loss_mask_multi = get_multimask(loss_mask)
        noise = self.randn_like(x_0)
        xt, alpha, sigma = self.q_sample(x_0, steps, noise)
        condition = self.get_denoiser_condition(steps)
        prediction, prediction_multi = self.denoiser(xt, condition)
        # x_denoise = (xt - sigma * prediction) / alpha
        # # plt.imshow(x_denoise[0,0].cpu().detach().numpy(), cmap="Greys")
        # # plt.savefig('./debug-fig/x_denoise.png', dpi=300, bbox_inches='tight', pad_inches=0)
        # _,(LHY,HLY,HHY) = pywt.dwt2(x_denoise[:,0].cpu().detach().numpy(), 'haar')
        # _,(LHY_0,HLY_0,HHY_0) = pywt.dwt2(x_0[:,0].cpu().detach().numpy(), 'haar')
        # loss_denoise = self.criterion(torch.tensor(LHY).unsqueeze(dim=1).to(device), torch.tensor(LHY_0).unsqueeze(dim=1).to(device))
        # + self.criterion(torch.tensor(HLY).unsqueeze(dim=1).to(device), torch.tensor(HLY_0).unsqueeze(dim=1).to(device)) 
        # + self.criterion(torch.tensor(HHY).unsqueeze(dim=1).to(device), torch.tensor(HHY_0).unsqueeze(dim=1).to(device))
        # loss_denoise = einops.reduce(loss_denoise * loss_mask_multi[0][:,0].unsqueeze(dim=1), "B ... -> B ()", "sum")
        # loss_mask_denoise = einops.reduce(loss_mask_multi[0][:,0].unsqueeze(dim=1), "B ... -> B ()", "sum")
        # loss_denoise = loss_denoise / loss_mask_denoise.add(1e-8)

        target = self.get_target(x_0, steps, noise)
        target_multi = get_multisize(target)

        loss = self.criterion(prediction, target)  # (B,C,H,W)
        loss = einops.reduce(loss * loss_mask, "B ... -> B ()", "sum")
        loss_mask = einops.reduce(loss_mask, "B ... -> B ()", "sum")
        loss = loss / loss_mask.add(1e-8)  # (B,)

        loss_2 = self.criterion(prediction_multi[0], target_multi[0])
        loss_2 = einops.reduce(loss_2 * loss_mask_multi[0], "B ... -> B ()", "sum")
        loss_mask_multi[0] = einops.reduce(loss_mask_multi[0], "B ... -> B ()", "sum")
        loss_2 = loss_2 / loss_mask_multi[0].add(1e-8)

        loss_4 = self.criterion(prediction_multi[1], target_multi[1])
        loss_4 = einops.reduce(loss_4 * loss_mask_multi[1], "B ... -> B ()", "sum")
        loss_mask_multi[1] = einops.reduce(loss_mask_multi[1], "B ... -> B ()", "sum")
        loss_4 = loss_4 / loss_mask_multi[1].add(1e-8)

        loss_8 = self.criterion(prediction_multi[2], target_multi[2])
        loss_8 = einops.reduce(loss_8 * loss_mask_multi[2], "B ... -> B ()", "sum")
        loss_mask_multi[2] = einops.reduce(loss_mask_multi[2], "B ... -> B ()", "sum")
        loss_8 = loss_8 / loss_mask_multi[2].add(1e-8)

        loss_16 = self.criterion(prediction_multi[3], target_multi[3])
        loss_16 = einops.reduce(loss_16 * loss_mask_multi[3], "B ... -> B ()", "sum")
        loss_mask_multi[3] = einops.reduce(loss_mask_multi[3], "B ... -> B ()", "sum")
        loss_16 = loss_16 / loss_mask_multi[3].add(1e-8)

        loss = ((loss_weights[0]*loss + loss_weights[1]*loss_2 + loss_weights[2]*loss_4 + loss_weights[3]*loss_8 + loss_weights[4]*loss_16)
                 * self.get_loss_weight(steps)).mean()
        return loss

    def forward(
        self, x_0: torch.Tensor, loss_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # shared in continuous/discrete versions
        steps = self.sample_timesteps(x_0.shape[0], x_0.device)
        loss = self.p_loss(x_0, steps, loss_mask)
        return loss

    @torch.inference_mode()
    def sample(self, *args, **kwargs):
        raise NotImplementedError
