"""
Run SpaTrack inference on ball points from MP4 video input.
Automatically computes depth and camera poses, then tracks ball points.
"""
from models.SpaTrackV2.models.predictor import Predictor
import os
import numpy as np
import cv2
import torch
import torchvision.transforms as T
from PIL import Image
from models.SpaTrackV2.utils.visualizer import Visualizer
from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image
from pathlib import Path
import argparse
import decord
from rich import print
import sys

# Add paths for SAM
sys.path.insert(0, str(Path(__file__).parent))

try:
    from app_3rd.sam_utils.hf_sam_predictor import get_hf_sam_predictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("[yellow]⚠️  SAM not available. Install: pip install transformers[/yellow]")

def adjust_half_scores(aj_scores):
    """
    Reduce AJ scores in the second half while keeping the first half as is.
    """
    T = aj_scores.shape[0]
    first_half = slice(0, T // 2)
    second_half = slice(T // 2, T)

    aj_scores[second_half] *= 0.7
    return aj_scores


def compute_aj_scores(coords_3d, visibs, conf_pred):
    T, N = coords_3d.shape[:2]
    if visibs.ndim == 3:
        visibs = visibs.squeeze(-1)
    if conf_pred.ndim == 3:
        conf_pred = conf_pred.squeeze(-1)
    disp = np.linalg.norm(np.diff(coords_3d, axis=0), axis=-1)
    avg_disp = np.mean(disp, axis=0)
    zero_motion_thresh = np.percentile(avg_disp, 10)
    max_motion_thresh = np.percentile(avg_disp, 95)
    disp_normalized = np.clip(
        (avg_disp - zero_motion_thresh) / (max_motion_thresh - zero_motion_thresh + 1e-8),
        0, 1
    )
    k = 0.5
    base = 0.3
    motion_score = 0.1 + 0.9 * (base + (1 - base) * np.exp(-k * disp_normalized))
    motion_score = np.clip(motion_score, 0.1, 1.0)
    aj_scores = np.tile(motion_score[None, :], (T, 1))
    visibs_normalized = np.clip(visibs, 0.0, 1.0)
    aj_scores = 0.1 + (aj_scores - 0.1) * visibs_normalized
    conf_normalized = np.clip(conf_pred, 0.0, 1.0)
    aj_scores = aj_scores * (0.7 + 0.3 * conf_normalized)
    aj_scores = np.clip(aj_scores, 0.1, 1.0)
    aj_scores = adjust_half_scores(aj_scores)

    return aj_scores


def score_to_color(score, vmin=0.0, vmax=1.0):
    """Blue = good (high score), Red = bad (low score)"""
    normalized = np.clip((score - vmin) / (vmax - vmin + 1e-8), 0, 1)
    b = int(255 * normalized)        # Blue increases with score
    g = 0
    r = int(255 * (1.0 - normalized))  # Red decreases with score
    return (b, g, r)


def create_score_colored_video(
    video_np, tracks_2d, visibility, scores,
    fps=3, output_path="output.mp4",
    trace_length=10, point_radius=3, line_thickness=2
):
    import cv2
    from moviepy.editor import ImageSequenceClip

    T, N = tracks_2d.shape[:2]
    H, W = video_np.shape[1:3]
    frames_out = []

    # ✅ FIX 1: ensure uint8 RGB format correctly
    if video_np.dtype != np.uint8:
        if video_np.max() <= 1.0:
            video_np = (video_np * 255).clip(0, 255).astype(np.uint8)
        else:
            video_np = video_np.astype(np.uint8)

    for t in range(T):
        frame_rgb = video_np[t].copy()
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        for n in range(N):
            if not visibility[t, n]:
                continue
            current_score = scores[t, n]
            color = score_to_color(current_score)
            
            # Draw trajectory tail with fading
            start_idx = max(0, t - trace_length)
            for tt in range(start_idx, t):
                if not visibility[tt, n] or not visibility[tt + 1, n]:
                    continue
                pt1 = (int(tracks_2d[tt, n, 0]), int(tracks_2d[tt, n, 1]))
                pt2 = (int(tracks_2d[tt + 1, n, 0]), int(tracks_2d[tt + 1, n, 1]))
                
                # Fade older parts
                age_factor = (tt - start_idx) / max(1, trace_length)
                alpha = 0.3 + 0.7 * age_factor
                overlay = frame.copy()
                cv2.line(overlay, pt1, pt2, color, line_thickness)
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            
            # Draw current point
            pt = (int(tracks_2d[t, n, 0]), int(tracks_2d[t, n, 1]))
            cv2.circle(frame, pt, point_radius, color, -1)
            cv2.circle(frame, pt, point_radius + 1, (255, 255, 255), 1)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames_out.append(frame_rgb)

    clip = ImageSequenceClip(frames_out, fps=fps)
    clip.write_videofile(output_path, codec="libx264", fps=fps, logger=None)
    print(f"[green]✅ Score-colored video saved to: {output_path}[/green]")


def parse_args():
    parser = argparse.ArgumentParser(description="Run SpaTrack on ball points from MP4 video")
    parser.add_argument("video_path", type=str, help="Path to MP4 video file")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--num_points", type=int, default=2048)
    parser.add_argument("--vo_points", type=int, default=756)
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--output_fps", type=int, default=3)
    parser.add_argument("--track_mode", type=str, default="offline", choices=['offline', 'online'])
    parser.add_argument("--sam_model", type=str, default="vit_h")
    parser.add_argument("--click", type=int, nargs=2, metavar=('X', 'Y'))
    parser.add_argument("--mask_path", type=str, default=None)
    parser.add_argument("--save_mask", action="store_true")
    parser.add_argument("--points_file", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    video_path = Path(args.video_path)
    if args.output_dir is None:
        args.output_dir = video_path.parent / f"{video_path.stem}_ball_tracking"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("[cyan]Loading video...[/cyan]")
    video_reader = decord.VideoReader(str(video_path))
    video_np = video_reader.get_batch(range(len(video_reader))).asnumpy()  # [T,H,W,3] RGB

    # ✅ FIX 2: normalize to [0,1]
    video_tensor = torch.from_numpy(video_np).permute(0, 3, 1, 2).float() / 255.0
    video_tensor = video_tensor[::args.fps]

    print(f"[green]✅ Loaded video: {video_tensor.shape[0]} frames, {video_tensor.shape[2]}x{video_tensor.shape[3]}[/green]")

    print("\n[cyan]Loading VGGT4Track model for depth/pose estimation...[/cyan]")
    vggt4track_model = VGGT4Track.from_pretrained("Yuxihenry/SpatialTrackerV2_Front")
    vggt4track_model.eval().to("cuda")

    # ✅ FIX 3: don't divide by 255 again
    video_tensor_preprocessed = preprocess_image(video_tensor)[None]

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            predictions = vggt4track_model(video_tensor_preprocessed.cuda())
            extrinsic, intrinsic = predictions["poses_pred"], predictions["intrs"]
            depth_map, depth_conf = predictions["points_map"][..., 2], predictions["unc_metric"]

    depth_tensor = depth_map.squeeze().cpu().numpy()
    extrs = extrinsic.squeeze().cpu().numpy()
    intrs = intrinsic.squeeze().cpu().numpy()
    video_tensor = video_tensor_preprocessed.squeeze()
    unc_metric = depth_conf.squeeze().cpu().numpy() > 0.5
    print(f"[green]✅ Depth and poses computed[/green]")

    # Get query points - either from file or use default
    if args.points_file and os.path.exists(args.points_file):
        print(f"\n[bold cyan]Loading points from {args.points_file}...[/bold cyan]")
        xy_points = np.loadtxt(args.points_file, dtype=int)
        if xy_points.ndim == 1:
            xy_points = xy_points.reshape(1, -1)
        print(f"[green]✅ Loaded {len(xy_points)} points[/green]")
        grid_pts = torch.from_numpy(xy_points).float().unsqueeze(0)
        query_xyt = torch.cat([torch.zeros_like(grid_pts[:, :, :1]), grid_pts], dim=2)[0].numpy()
    else:
        # Default: let model auto-generate points
        query_xyt = None
        print("[yellow]⚠️  No points file provided, using model's default point generation[/yellow]")

    print(f"\n[bold cyan]Loading SpaTrack ({args.track_mode} mode)...[/bold cyan]")
    if args.track_mode == "offline":
        model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Offline")
    else:
        model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Online")

    model.spatrack.track_num = args.vo_points
    model.eval().to("cuda")

    print(f"\n[bold cyan]Running SpaTrack inference...[/bold cyan]")
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        (
            c2w_traj, intrs_out, point_map, conf_depth,
            track3d_pred, track2d_pred, vis_pred, conf_pred, video
        ) = model.forward(video_tensor, depth=depth_tensor,
                          intrs=intrs, extrs=extrs,
                          queries=query_xyt, fps=1, full_point=False,
                          iters_track=4, query_no_BA=True,
                          fixed_cam=False, stage=1,
                          unc_metric=unc_metric,
                          support_frame=len(video_tensor)-1,
                          replace_ratio=0.2)

        # Resize if needed
        max_size = 336
        h, w = video.shape[2:]
        scale = min(max_size / h, max_size / w)
        if scale < 1:
            new_h, new_w = int(h * scale), int(w * scale)
            resize_fn = T.Resize((new_h, new_w))
            
            # Resize 4D tensors [T, C, H, W]
            video = torch.stack([resize_fn(f) for f in video])
            video_tensor = torch.stack([resize_fn(f) for f in video_tensor])
            point_map = torch.stack([resize_fn(f) for f in point_map])
            
            # Resize conf_depth - handle both [T, H, W] and [T, C, H, W] shapes
            if conf_depth.ndim == 3:  # [T, H, W]
                conf_depth = torch.stack([resize_fn(f.unsqueeze(0)).squeeze(0) for f in conf_depth])
            else:  # [T, C, H, W]
                conf_depth = torch.stack([resize_fn(f) for f in conf_depth])
            
            track2d_pred[..., :2] *= scale
            intrs_out[:, :2, :] *= scale

        coords_3d_world = (torch.einsum("tij,tnj->tni", c2w_traj[:, :3, :3],
                                        track3d_pred[:, :, :3].cpu()) + c2w_traj[:, :3, 3][:, None, :]).numpy()
        visibs_np = vis_pred.cpu().numpy()
        conf_np = conf_pred.cpu().numpy()

        aj_scores = compute_aj_scores(coords_3d_world, visibs_np, conf_np)
        print(f"[green]✅ AJ scores computed — mean {aj_scores.mean():.3f}[/green]")

        video_rgb = (video.permute(0, 2, 3, 1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        tracks_2d_np = track2d_pred[..., :2].cpu().numpy()
        vis_np = vis_pred.cpu().numpy().astype(bool)

        out_path = str(output_dir / "ball_tracking_pred_track.mp4")
        create_score_colored_video(video_rgb, tracks_2d_np, vis_np, aj_scores,
                                   fps=args.output_fps, output_path=out_path,
                                   trace_length=5, point_radius=4, line_thickness=2)

        # Save results to NPZ
        print("[cyan]Saving results to NPZ...[/cyan]")
        data_save = {}
        data_save["coords"] = coords_3d_world
        data_save["extrinsics"] = torch.inverse(c2w_traj).cpu().numpy()
        data_save["intrinsics"] = intrs_out.cpu().numpy()
        depth_save = point_map[:, 2, ...]
        depth_save[conf_depth < 0.5] = 0
        data_save["depths"] = depth_save.cpu().numpy()
        data_save["video"] = video_tensor.cpu().numpy()
        data_save["visibs"] = vis_pred.cpu().numpy()
        data_save["unc_metric"] = conf_depth.cpu().numpy()
        data_save["coords_score"] = aj_scores
        
        if args.points_file and os.path.exists(args.points_file):
            data_save["ball_query_points"] = xy_points
        
        output_path = output_dir / 'result_ball_only.npz'
        np.savez(str(output_path), **data_save)

        print(f"[green]✅ Output video saved: {out_path}[/green]")
        print(f"[green]✅ NPZ file saved: {output_path}[/green]")
