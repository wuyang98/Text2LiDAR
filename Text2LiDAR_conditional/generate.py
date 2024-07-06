import argparse
from pathlib import Path
from models.CLIP.clip import clip
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
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True
    clip_model = clip.load("ViT-B/32",device=args.device)
    # =================================================================================
    # Load pre-trained model
    # =================================================================================

    ddpm, lidar_utils, _ = utils.inference.setup_model(args.ckpt, device=args.device)

    # =================================================================================
    # Sampling (reverse diffusion)
    # =================================================================================
    # text = ["Construction, maneuver between several trucks"] # 1
    # text = ["Surrounded by buildings"] # 2
    # text = ["Cross busy intersection, cyclist nearby, construction barriers and cones"] # 3
    # text = ["Drive between buildings under construction"] # 4
    # text = ["Rain, parked cars, construction"] # 5
    # text = ["Rain, long street, parked car", "Rain, long street, parked car", "Rain, long street, parked car", "Rain, long street, parked car"] # 6
    # text = ["Rain, long street, parked car", "Rain, long street, parked car", "long street, parked car", "long street, parked car"] # 7
    # text = ["Rain, long street, parked car", "Night, long street, parked car", "Foggy, long street, parked car", "long street, parked car"] # 8
    # text = ["Rain, heavy rain, very heavy rain, behind heavy truck", "Behind heavy truck"] # 9
    # text = ["In front of heavy truck", "Behind heavy truck", "There are buildings on both sides", "There are buildings in front and behind"] # 10
    # text = ["Dense traffic at other lane", "Bus, turn right, peds, bus stop"] # 11
    # text = ["Cross intersection, construction vehicle, truck, many peds", "Cross intersection, construction vehicle, truck, many peds"] # 12
    # text = ["Rain, Cross intersection, construction vehicle, truck, many peds", "Cross intersection, construction vehicle, truck, many peds"] # 13
    # text = ["Rain", "Wet ground"] # 14 
    # text = ["Rain, long street, parked car", "Rain, long street, parked car", "long street, parked car", "long street, parked car"] # 15
    # text = ["parked cars", "parked cars", "parked cars", "parked cars"] # 16
    text = ["Heavy rain"] # 17


    text_emb = clip.tokenize(text).to(args.device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_emb) # B, 512
    timestamp = time.time()
    local_time = time.ctime(timestamp)
    print("本地时间：",timestamp)
    xs = ddpm.sample(
        batch_size=args.batch_size,
        num_steps=args.sampling_steps,
        return_all=True,
        text=text_features,
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
        # R, t = utils.render.make_Rt(pitch=0, yaw=0, z=0.8)
        bev = 1 - utils.render.render_point_clouds(
            points=einops.rearrange(xyz, "B C H W -> B (H W) C"),
            colors=1 - einops.rearrange(colors, "B C H W -> B (H W) C"),
            R=R.to(xyz),
            t=t.to(xyz),
        )
        return img, bev
    
    img, bev = render(xs[-1])
    save_image(img, "/project/r2dm-main/logs/diffusion/nuScenes/spherical-1024/forpaper/results/samples_img.png", nrow=1)
    save_image(bev, "/project/r2dm-main/logs/diffusion/nuScenes/spherical-1024/forpaper/results/samples_bev.png", nrow=4)

    video = imageio.get_writer("/project/r2dm-main/logs/diffusion/nuScenes/spherical-1024/forpaper/results/samples.mp4", mode="I", fps=60)
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
    parser.add_argument("--ckpt", type=Path, default='/project/r2dm-main/logs/diffusion/nuScenes/spherical-1024/forpaper/models/diffusion_0000400000.pth')
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--sampling_steps", type=int, default=128)
    args = parser.parse_args()
    args.device = torch.device(args.device)
    main(args)
