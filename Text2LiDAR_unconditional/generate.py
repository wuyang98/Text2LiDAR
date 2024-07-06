import argparse
from pathlib import Path

import einops
import imageio
import matplotlib.cm as cm
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm
import time
import utils.inference
import utils.render
import matplotlib.pyplot as plt


def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True

    # =================================================================================
    # Load pre-trained model
    # =================================================================================

    ddpm, lidar_utils, _ = utils.inference.setup_model(args.ckpt, device=args.device)

    # =================================================================================
    # Sampling (reverse diffusion)
    # =================================================================================
    timestamp = time.time()
    local_time = time.ctime(timestamp)
    print("本地时间：",timestamp)
    xs = ddpm.sample(
        batch_size=args.batch_size,
        num_steps=args.sampling_steps,
        return_all=True,
    ).clamp(-1, 1)
    timestamp = time.time()
    local_time = time.ctime(timestamp)
    print("本地时间：",timestamp)
    # =================================================================================
    # Save as image or video
    # =================================================================================

    xs = lidar_utils.denormalize(xs)
    xs[:, :, [0]] = lidar_utils.revert_depth(xs[:, :, [0]]) / lidar_utils.max_depth

    def render(x):
        img = einops.rearrange(x, "B C H W -> B 1 (C H) W")
        img = utils.render.colorize(img) / 255
        xyz = lidar_utils.to_xyz(x[:, [0]] * lidar_utils.max_depth)
        xyz /= lidar_utils.max_depth
        z_min, z_max = -2 / lidar_utils.max_depth, 0.5 / lidar_utils.max_depth
        z = (xyz[:, [2]] - z_min) / (z_max - z_min)
        colors = utils.render.colorize(z.clamp(0, 1), cm.viridis) / 255
        R, t = utils.render.make_Rt(pitch=torch.pi / 3, yaw=torch.pi / 4, z=0.8)
        bev = 1 - utils.render.render_point_clouds(
            points=einops.rearrange(xyz, "B C H W -> B (H W) C"),
            colors=1 - einops.rearrange(colors, "B C H W -> B (H W) C"),
            R=R.to(xyz),
            t=t.to(xyz),
        )
        return img, bev
    
    img, bev = render(xs[-1])
    save_image(img, "/project/r2dm-transformer-5decoder-dwt/forsupp_img.png", nrow=1)
    save_image(bev, "/project/r2dm-transformer-5decoder-dwt/forsupp_bev.png", nrow=4)

    video = imageio.get_writer("/project/r2dm-transformer-5decoder-dwt/samples.mp4", mode="I", fps=60)
    t = 0
    for x in tqdm(xs, desc="making video..."):
        t = t + 1
        img, bev = render(x)
        scale = 512 / img.shape[-1]
        img = F.interpolate(img, scale_factor=scale, mode="bilinear", antialias=True)
        scale = 512 / bev.shape[-1]
        bev = F.interpolate(bev, scale_factor=scale, mode="bilinear", antialias=True)
        img = torch.cat([img, bev], dim=2)
        img = make_grid(img, nrow=args.batch_size, pad_value=1)
        img = img.permute(1, 2, 0).mul(255).byte()
        video.append_data(img.cpu().numpy())
        if t == 257:
            for tt in range(100):
                video.append_data(img.cpu().numpy())
    video.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default='/project/r2dm-transformer-5decoder-dwt/logs/diffusion/kitti_360/spherical-1024/dwt-convpos/models/diffusion_0000300000.pth')
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--sampling_steps", type=int, default=256)
    parser.add_argument("--seed", type=int, default=516) # can be used: 3
    args = parser.parse_args()
    args.device = torch.device(args.device)
    main(args)
