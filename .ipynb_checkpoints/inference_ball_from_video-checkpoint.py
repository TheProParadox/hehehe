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


def compute_aj_scores(coords_3d, visibs, conf_pred):
    """
    Compute AJ (Average Jaccard) scores based on 3D tracking quality.
    
    Args:
        coords_3d: numpy array [T, N, 3] - 3D world coordinates
        visibs: numpy array [T, N] or [T, N, 1] - visibility predictions
        conf_pred: numpy array [T, N] or [T, N, 1] - confidence predictions
    
    Returns:
        aj_scores: numpy array [T, N] - AJ scores in [0, 1]
    """
    T, N = coords_3d.shape[:2]
    
    # Ensure visibs and conf_pred are [T, N]
    if visibs.ndim == 3:
        visibs = visibs.squeeze(-1)
    if conf_pred.ndim == 3:
        conf_pred = conf_pred.squeeze(-1)
    
    # Compute 3D displacement magnitudes
    disp = np.linalg.norm(np.diff(coords_3d, axis=0), axis=-1)  # [T-1, N]
    avg_disp = np.mean(disp, axis=0)  # [N]
    
    # Normalize motion range (points with more motion get lower base scores)
    zero_motion_thresh = np.percentile(avg_disp, 10)
    max_motion_thresh = np.percentile(avg_disp, 95)
    disp_normalized = np.clip(
        (avg_disp - zero_motion_thresh) / (max_motion_thresh - zero_motion_thresh + 1e-8),
        0, 1
    )
    
    # Base score from motion (gentler exponential decay)
    k = 0.5  # decay rate
    base = 0.3  # minimum contribution
    motion_score = 0.1 + 0.9 * (base + (1 - base) * np.exp(-k * disp_normalized))
    motion_score = np.clip(motion_score, 0.1, 1.0)
    
    # Broadcast to all frames [T, N]
    aj_scores = np.tile(motion_score[None, :], (T, 1))
    
    # Weight by visibility
    visibs_normalized = np.clip(visibs, 0.0, 1.0)
    aj_scores = 0.1 + (aj_scores - 0.1) * visibs_normalized
    
    # Weight by confidence (give it less weight than visibility)
    conf_normalized = np.clip(conf_pred, 0.0, 1.0)
    aj_scores = aj_scores * (0.7 + 0.3 * conf_normalized)
    
    # Final clipping
    aj_scores = np.clip(aj_scores, 0.1, 1.0)
    
    return aj_scores


def score_to_color(score, vmin=0.0, vmax=1.0):
    normalized = np.clip((score - vmin) / (vmax - vmin + 1e-8), 0, 1)
    
    # High = blue, low = red
    b = int(255 * (1.0 - normalized))  # blue decreases as score increases
    g = 0
    r = int(255 * normalized)          # red increases with score

    return (b, g, r)



def create_score_colored_video(
    video_np, tracks_2d, visibility, scores,
    fps=3, output_path="output.mp4",
    trace_length=10, point_radius=3, line_thickness=2
):
    """
    Create a video with AJ score-based coloring overlayed on the original RGB video.
    """
    import cv2
    from moviepy.editor import ImageSequenceClip

    T, N = tracks_2d.shape[:2]
    H, W = video_np.shape[1:3]
    frames_out = []

    # --- Ensure correct format and range ---
    if video_np.dtype != np.uint8:
        if video_np.max() <= 1.0:
            video_np = (video_np * 255).astype(np.uint8)
        else:
            video_np = video_np.astype(np.uint8)

    # --- Iterate through frames ---
    for t in range(T):
        # Start with the original frame in RGB
        frame_rgb = video_np[t].copy()

        # Convert to BGR *only for OpenCV drawing*
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # --- Draw trajectories ---
        for n in range(N):
            if not visibility[t, n]:
                continue

            current_score = scores[t, n]
            color = score_to_color(current_score)  # (B, G, R)

            # Draw short trajectory tail
            start_idx = max(0, t - trace_length)
            for tt in range(start_idx, t):
                if not visibility[tt, n] or not visibility[tt + 1, n]:
                    continue
                pt1 = (int(tracks_2d[tt, n, 0]), int(tracks_2d[tt, n, 1]))
                pt2 = (int(tracks_2d[tt + 1, n, 0]), int(tracks_2d[tt + 1, n, 1]))
                cv2.line(frame, pt1, pt2, color, line_thickness)

            # Draw current point
            pt = (int(tracks_2d[t, n, 0]), int(tracks_2d[t, n, 1]))
            cv2.circle(frame, pt, point_radius, color, -1)
            cv2.circle(frame, pt, point_radius + 1, (255, 255, 255), 1)

        # Convert back to RGB for MoviePy
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames_out.append(frame_rgb)

    # --- Save video ---
    clip = ImageSequenceClip(frames_out, fps=fps)
    clip.write_videofile(output_path, codec="libx264", fps=fps, logger=None)
    print(f"[green]✅ Score-colored video saved to: {output_path}[/green]")

def segment_ball_with_sam(image, model_type='vit_h', device=None):
    """Segment ball using SAM with interactive selection."""
    if not SAM_AVAILABLE:
        raise ImportError("SAM not available. Install: pip install transformers")
    
    print(f"[cyan]Loading SAM model ({model_type})...[/cyan]")
    predictor = get_hf_sam_predictor(model_type=model_type, device=device)
    
    print("[cyan]Running automatic segmentation...[/cyan]")
    H, W = image.shape[:2]
    
    # Create a grid of points across the image
    grid_size = 32
    x_points = np.linspace(W//4, 3*W//4, grid_size)
    y_points = np.linspace(H//4, 3*H//4, grid_size)
    xx, yy = np.meshgrid(x_points, y_points)
    points = np.stack([xx.flatten(), yy.flatten()], axis=1)
    
    # Process with SAM
    input_points = [points.tolist()]
    input_labels = [[1] * len(points)]
    
    inputs = predictor.preprocess(image, input_points, input_labels)
    
    # Get predictions
    with torch.no_grad():
        outputs = predictor.model(**inputs)
    
    # Get masks
    masks = predictor.processor.post_process_masks(
        outputs.pred_masks,
        inputs['original_sizes'],
        inputs['reshaped_input_sizes']
    )[0]
    
    masks = masks.cpu().numpy()
    scores = outputs.iou_scores.cpu().numpy().flatten()
    
    print(f"[cyan]Generated {len(masks)} masks[/cyan]")
    
    # Interactive selection
    selected_mask = interactive_segment_selection(image, masks, scores)
    
    return selected_mask


def interactive_segment_selection(image, masks, scores):
    """Interactive UI to select the ball segment."""
    sorted_indices = np.argsort(scores)[::-1]
    
    print("\n[bold cyan]═══ Interactive Ball Segment Selection ═══[/bold cyan]")
    print("[yellow]Instructions:[/yellow]")
    print("  • Press [bold]'n'[/bold] for next segment")
    print("  • Press [bold]'p'[/bold] for previous segment")
    print("  • Press [bold]'s'[/bold] to select current segment (the ball)")
    print("  • Press [bold]'q'[/bold] to quit without selection")
    print("[bold cyan]════════════════════════════════════════[/bold cyan]\n")
    
    current_idx = 0
    selected_mask = None
    
    while True:
        idx = sorted_indices[current_idx]
        mask = masks[idx, 0] > 0
        score = scores[idx]
        
        # Create visualization
        vis_image = image.copy()
        overlay = vis_image.copy()
        overlay[mask] = [0, 255, 0]
        vis_image = cv2.addWeighted(vis_image, 0.6, overlay, 0.4, 0)
        
        # Add border
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_image, contours, -1, (0, 255, 0), 3)
        
        # Add text
        text = f"Segment {current_idx + 1}/{len(masks)} | Score: {score:.3f} | Press 's' to select"
        cv2.putText(vis_image, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
        cv2.putText(vis_image, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        cv2.imshow('Select Ball Segment', cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('n'):
            current_idx = (current_idx + 1) % len(masks)
        elif key == ord('p'):
            current_idx = (current_idx - 1) % len(masks)
        elif key == ord('s'):
            selected_mask = mask
            print(f"[bold green]✅ Selected segment {current_idx + 1} with score {score:.3f}[/bold green]")
            break
        elif key == ord('q'):
            print("[bold red]❌ Selection cancelled[/bold red]")
            break
    
    cv2.destroyAllWindows()
    return selected_mask


def segment_ball_with_click(image, click_point, model_type='vit_h', device=None):
    """Segment ball at clicked point using SAM."""
    if not SAM_AVAILABLE:
        raise ImportError("SAM not available. Install: pip install transformers")
    
    print(f"[cyan]Loading SAM model ({model_type})...[/cyan]")
    predictor = get_hf_sam_predictor(model_type=model_type, device=device)
    
    print(f"[cyan]Segmenting ball at point {click_point}...[/cyan]")
    
    input_points = [[click_point]]
    input_labels = [[1]]
    
    inputs = predictor.preprocess(image, input_points, input_labels)
    
    with torch.no_grad():
        outputs = predictor.model(**inputs)
    
    masks_raw = predictor.processor.post_process_masks(
        outputs.pred_masks,
        inputs['original_sizes'],
        inputs['reshaped_input_sizes']
    )
    
    # Get first batch and convert to numpy immediately
    masks = masks_raw[0]
    if hasattr(masks, 'cpu'):
        masks_np = masks.cpu().numpy()
    else:
        masks_np = np.array(masks)
    
    scores = outputs.iou_scores.cpu().numpy().flatten()
    best_idx = np.argmax(scores)
    
    # Debug output
    print(f"[cyan]Debug: masks shape: {masks_np.shape}, best_idx: {best_idx}, num_scores: {len(scores)}[/cyan]")
    
    # Handle different mask shapes
    if masks_np.ndim == 4:  # [batch, num_masks, H, W]
        # Shape is [1, 3, 512, 512] -> index as [0, best_idx] to get [H, W]
        mask = masks_np[0, best_idx] > 0
    elif masks_np.ndim == 3:  # [N, H, W] or [1, H, W]
        if masks_np.shape[0] == 1:  # [1, H, W] - single batch, single mask
            mask = masks_np[0] > 0
        else:  # [N, H, W] - multiple masks
            mask = masks_np[best_idx] > 0
    elif masks_np.ndim == 2:  # [H, W] - already squeezed
        mask = masks_np > 0
    else:
        raise ValueError(f"Unexpected mask shape: {masks_np.shape}")
    
    print(f"[green]✅ Segmentation complete! Mask shape: {mask.shape}, pixels: {mask.sum()}[/green]")
    
    return mask


def parse_args():
    parser = argparse.ArgumentParser(description="Run SpaTrack on ball points from MP4 video")
    parser.add_argument("video_path", type=str, help="Path to MP4 video file")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--num_points", type=int, default=2048, help="Number of points to track on ball")
    parser.add_argument("--vo_points", type=int, default=756, help="VO points for SpaTrack")
    parser.add_argument("--fps", type=int, default=1, help="Frame sampling rate (1=every frame, 2=every 2nd frame, etc)")
    parser.add_argument("--output_fps", type=int, default=3, help="Output video FPS")
    parser.add_argument("--track_mode", type=str, default="offline", choices=['offline', 'online'])
    
    # SAM options
    parser.add_argument("--sam_model", type=str, default="vit_h", choices=['vit_b', 'vit_l', 'vit_h'])
    parser.add_argument("--click", type=int, nargs=2, metavar=('X', 'Y'),
                       help="Click point (x, y) to segment ball directly")
    parser.add_argument("--mask_path", type=str, default=None,
                       help="Path to pre-saved ball mask")
    parser.add_argument("--save_mask", action="store_true", help="Save ball mask for reuse")
    
    # Direct point selection (skip SAM)
    parser.add_argument("--points_file", type=str, default=None,
                       help="Path to text file with points (x y per line) - skips SAM segmentation")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Setup output directory
    video_path = Path(args.video_path)
    if args.output_dir is None:
        args.output_dir = video_path.parent / f"{video_path.stem}_ball_tracking"
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("[bold cyan]═══════════════════════════════════════════[/bold cyan]")
    print("[bold cyan]   SpaTrack Ball Tracking from Video[/bold cyan]")
    print("[bold cyan]═══════════════════════════════════════════[/bold cyan]\n")
    
    print(f"[cyan]Input video: {video_path}[/cyan]")
    print(f"[cyan]Output dir: {output_dir}[/cyan]\n")
    
    # Load video
    print("[cyan]Loading video...[/cyan]")
    video_reader = decord.VideoReader(str(video_path))
    video_tensor = torch.from_numpy(video_reader.get_batch(range(len(video_reader))).asnumpy()).permute(0, 3, 1, 2)
    video_tensor = video_tensor[::args.fps].float()
    
    print(f"[green]✅ Loaded video: {video_tensor.shape[0]} frames, {video_tensor.shape[2]}x{video_tensor.shape[3]}[/green]")
    
    # Get first frame for segmentation
    first_frame = video_tensor[0].permute(1, 2, 0).numpy().astype(np.uint8)
    frame_H, frame_W = first_frame.shape[:2]
    
    # Load VGGT4Track for depth and pose estimation
    print("\n[cyan]Loading VGGT4Track model for depth/pose estimation...[/cyan]")
    vggt4track_model = VGGT4Track.from_pretrained("Yuxihenry/SpatialTrackerV2_Front")
    vggt4track_model.eval()
    vggt4track_model = vggt4track_model.to("cuda")
    
    # Compute depth and poses
    print("[cyan]Computing depth and camera poses...[/cyan]")
    video_tensor_preprocessed = preprocess_image(video_tensor)[None]
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            predictions = vggt4track_model(video_tensor_preprocessed.cuda()/255)
            extrinsic, intrinsic = predictions["poses_pred"], predictions["intrs"]
            depth_map, depth_conf = predictions["points_map"][..., 2], predictions["unc_metric"]
    
    depth_tensor = depth_map.squeeze().cpu().numpy()
    extrs = extrinsic.squeeze().cpu().numpy()
    intrs = intrinsic.squeeze().cpu().numpy()
    video_tensor = video_tensor_preprocessed.squeeze()
    unc_metric = depth_conf.squeeze().cpu().numpy() > 0.5
    
    print(f"[green]✅ Depth and poses computed[/green]")
    
    # Get query points - either from file or via SAM segmentation
    if args.points_file:
        # Direct point selection - skip SAM entirely
        print(f"\n[bold cyan]Loading points from {args.points_file}...[/bold cyan]")
        xy_points = np.loadtxt(args.points_file, dtype=int)
        
        if xy_points.ndim == 1:
            xy_points = xy_points.reshape(1, -1)
        
        print(f"[green]✅ Loaded {len(xy_points)} points (no SAM segmentation)[/green]")
        
        # No mask in this mode
        mask = None
        
    else:
        # SAM-based segmentation
        # Get ball mask
        if args.mask_path and os.path.exists(args.mask_path):
            print(f"[cyan]Loading pre-saved mask from {args.mask_path}...[/cyan]")
            mask = cv2.imread(args.mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (frame_W, frame_H))
            mask = mask > 127
        else:
            if not SAM_AVAILABLE:
                print("[bold red]❌ SAM not available and no mask provided![/bold red]")
                print("[yellow]Install SAM: pip install transformers torch[/yellow]")
                sys.exit(1)
            
            if args.click:
                mask = segment_ball_with_click(first_frame, args.click, args.sam_model, 'cuda')
            else:
                mask = segment_ball_with_sam(first_frame, args.sam_model, 'cuda')
            
            if mask is None:
                print("[bold red]❌ No mask selected. Exiting.[/bold red]")
                sys.exit(1)
            
            if args.save_mask:
                mask_save_path = output_dir / "ball_mask.png"
                cv2.imwrite(str(mask_save_path), (mask * 255).astype(np.uint8))
                print(f"[green]✅ Mask saved: {mask_save_path}[/green]")
        
        # Sample points on ball
        print(f"\n[bold cyan]Sampling {args.num_points} points on ball...[/bold cyan]")
        np.random.seed(42)
        valid_mask = mask.astype(bool)
        ys, xs = np.where(valid_mask)
        num_valid = len(xs)
        
        if num_valid < args.num_points:
            print(f"[yellow]⚠️  Only {num_valid} valid pixels; using all.[/yellow]")
            selected = np.arange(num_valid)
        else:
            selected = np.random.choice(num_valid, size=args.num_points, replace=False)
        
        xy_points = np.stack([xs[selected], ys[selected]], axis=1)
        print(f"[green]✅ Sampled {len(xy_points)} points on ball[/green]")
    
    # Convert points to query format
    grid_pts = torch.from_numpy(xy_points).float().unsqueeze(0)
    query_xyt = torch.cat([torch.zeros_like(grid_pts[:, :, :1]), grid_pts], dim=2)[0].numpy()
    
    print(f"[green]✅ Total tracking points: {len(xy_points)}[/green]")
    
    # Visualize sampled points
    vis_points = first_frame.copy()
    for x, y in xy_points:
        cv2.circle(vis_points, (int(x), int(y)), 2, (0, 255, 0), -1)
    cv2.imwrite(str(output_dir / "ball_points_visualization.png"), 
                cv2.cvtColor(vis_points, cv2.COLOR_RGB2BGR))
    print(f"[green]✅ Point visualization: {output_dir}/ball_points_visualization.png[/green]")
    
    # Load SpaTrack
    print(f"\n[bold cyan]Loading SpaTrack ({args.track_mode} mode)...[/bold cyan]")
    if args.track_mode == "offline":
        model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Offline")
    else:
        model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Online")
    
    model.spatrack.track_num = args.vo_points
    model.eval()
    model.to("cuda")
    
    viser = Visualizer(save_dir=str(output_dir), grayscale=False, fps=args.output_fps, 
                       pad_value=0, tracks_leave_trace=5)
    
    # Run inference
    print(f"\n[bold cyan]Running SpaTrack inference...[/bold cyan]")
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        (
            c2w_traj, intrs_out, point_map, conf_depth,
            track3d_pred, track2d_pred, vis_pred, conf_pred, video
        ) = model.forward(video_tensor, depth=depth_tensor,
                            intrs=intrs, extrs=extrs, 
                            queries=query_xyt,
                            fps=1, full_point=False, iters_track=4,
                            query_no_BA=True, fixed_cam=False, stage=1, 
                            unc_metric=unc_metric,
                            support_frame=len(video_tensor)-1, replace_ratio=0.2)
        
        # Resize if needed
        max_size = 336
        h, w = video.shape[2:]
        scale = min(max_size / h, max_size / w)
        if scale < 1:
            new_h, new_w = int(h * scale), int(w * scale)
            video = T.Resize((new_h, new_w))(video)
            video_tensor = T.Resize((new_h, new_w))(video_tensor)
            point_map = T.Resize((new_h, new_w))(point_map)
            conf_depth = T.Resize((new_h, new_w))(conf_depth)
            track2d_pred[...,:2] = track2d_pred[...,:2] * scale
            intrs_out[:,:2,:] = intrs_out[:,:2,:] * scale
            if depth_tensor is not None:
                if isinstance(depth_tensor, torch.Tensor):
                    depth_tensor = T.Resize((new_h, new_w))(depth_tensor)
                else:
                    depth_tensor = T.Resize((new_h, new_w))(torch.from_numpy(depth_tensor))
        
        # Compute AJ scores
        print("[cyan]Computing AJ scores...[/cyan]")
        coords_3d_world = (torch.einsum("tij,tnj->tni", c2w_traj[:,:3,:3], 
                                        track3d_pred[:,:,:3].cpu()) + c2w_traj[:,:3,3][:,None,:]).numpy()
        visibs_np = vis_pred.cpu().numpy()
        conf_np = conf_pred.cpu().numpy()
        
        aj_scores = compute_aj_scores(coords_3d_world, visibs_np, conf_np)
        print(f"[green]✅ AJ scores computed - Mean: {aj_scores.mean():.4f}, Min: {aj_scores.min():.4f}, Max: {aj_scores.max():.4f}[/green]")

        # ---- AJ score summaries (add right after aj_scores is computed) ----
        per_frame_mean = aj_scores.mean(axis=1)   # [T] average across points for each frame
        per_point_mean = aj_scores.mean(axis=0)   # [N] average across frames for each point
        overall_mean   = per_frame_mean.mean()    # scalar
        
        print(f"[bold magenta]AJ per-frame mean (first 10 frames):[/bold magenta] {per_frame_mean[:10].round(4)}")
        print(f"[bold magenta]AJ per-frame stats:[/bold magenta]  "
              f"min={per_frame_mean.min():.4f}  p25={np.percentile(per_frame_mean,25):.4f}  "
              f"median={np.median(per_frame_mean):.4f}  p75={np.percentile(per_frame_mean,75):.4f}  "
              f"max={per_frame_mean.max():.4f}  overall_mean={overall_mean:.4f}")

        
        # Create score-colored visualization
        print("[cyan]Creating AJ score-colored visualization...[/cyan]")
        print("[dim](Blue = high AJ score/good tracking, Red = low AJ score/bad tracking)[/dim]")
        filename_base = "manual_points" if args.points_file else "ball_tracking"
        video_rgb = (video.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        tracks_2d_np = track2d_pred[..., :2].cpu().numpy()
        vis_np = vis_pred.cpu().numpy().astype(bool)
        
        score_video_path = str(output_dir / f"{filename_base}_pred_track.mp4")
        create_score_colored_video(
            video_rgb, 
            tracks_2d_np, 
            vis_np, 
            aj_scores,
            fps=args.output_fps,
            output_path=score_video_path,
            trace_length=5,
            point_radius=4,
            line_thickness=2
        )
        
        # Save results
        print("[cyan]Saving results...[/cyan]")
        data_save = {}
        data_save["coords"] = coords_3d_world
        data_save["extrinsics"] = torch.inverse(c2w_traj).cpu().numpy()
        data_save["intrinsics"] = intrs_out.cpu().numpy()
        depth_save = point_map[:,2,...]
        depth_save[conf_depth<0.5] = 0
        data_save["depths"] = depth_save.cpu().numpy()
        data_save["video"] = (video_tensor).cpu().numpy()/255
        data_save["visibs"] = vis_pred.cpu().numpy()
        data_save["unc_metric"] = conf_depth.cpu().numpy()
        if mask is not None:
            data_save["ball_mask"] = mask
        data_save["ball_query_points"] = xy_points
        data_save["coords_score"] = aj_scores  # Use AJ scores instead of raw confidence
        
        output_filename = 'result_manual_points.npz' if args.points_file else 'result_ball_only.npz'
        output_path = output_dir / output_filename
        np.savez(str(output_path), **data_save)
    
    mode_str = "Manual Points" if args.points_file else "Ball Tracking"
    print(f"\n[bold green]{'═'*50}[/bold green]")
    print(f"[bold green]✅ {mode_str} complete![/bold green]")
    print(f"[bold green]{'═'*50}[/bold green]")
    print(f"\n[bold]Results saved to:[/bold] [cyan]{output_dir}[/cyan]")
    print(f"  • NPZ file: [cyan]{output_path}[/cyan]")
    print(f"    [dim]Contains AJ scores (Average Jaccard) in 'coords_score'[/dim]")
    print(f"  • [bold cyan]AJ score-colored video: {output_dir}/{filename_base}_pred_track.mp4[/bold cyan]")
    print(f"    [dim](Blue = high AJ score/good tracking, Red = low AJ score/bad tracking)[/dim]")
    print(f"\n[bold yellow]To create side-by-side GIF with viz.html-style rendering:[/bold yellow]")
    print(f"  [cyan]python examples/results/visualize_tracks.py {output_path}[/cyan]\n")

