import json
import os

import cv2
import evo
import numpy as np
import torch
from evo.core import metrics, trajectory
from evo.core.metrics import PoseRelation, Unit
from evo.core.trajectory import PosePath3D, PoseTrajectory3D
from evo.tools import plot
from evo.tools.plot import PlotMode
from evo.tools.settings import SETTINGS
import matplotlib

matplotlib.use("Agg")  # Use a non-interactive backend for matplotlib
from matplotlib import pyplot as plt
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm

import wandb
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.image_utils import psnr
from gaussian_splatting.utils.loss_utils import ssim
from gaussian_splatting.utils.system_utils import mkdir_p
from utils.logging_utils import Log


def evaluate_evo(poses_gt, poses_est, result_dir, label, monocular=False):
    if len(poses_gt) < 5 or len(poses_est) < 5:
        ape_stat = 0.0
        rpe_stat = 0.0
        ape_stats = {}
        rpe_stats = {}
        Log("RMSE ATE [m]", ape_stat, tag="Eval")
        Log("RMSE RPE [m]", rpe_stat, tag="Eval")
        with open(
                os.path.join(result_dir, "stats_{}.json".format(str(label))),
                "w",
                encoding="utf-8",
        ) as f:
            json.dump({"ape": ape_stats, "rpe": rpe_stats}, f, indent=4)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(f"ATE RMSE: {ape_stat} RPE RMSE: {rpe_stat} (Insufficient poses)")
        plt.savefig(os.path.join(result_dir, "traj_{}.png".format(str(label))), dpi=90)
        return ape_stat, rpe_stat

    traj_ref = PosePath3D(poses_se3=poses_gt)
    traj_est = PosePath3D(poses_se3=poses_est)
    traj_est_aligned = trajectory.align_trajectory(traj_est, traj_ref, correct_scale=monocular)

    pose_relation = metrics.PoseRelation.translation_part
    data = (traj_ref, traj_est_aligned)

    ape_metric = metrics.APE(pose_relation)
    ape_metric.process_data(data)
    ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)
    ape_stats = ape_metric.get_all_statistics()

    rpe_metric = metrics.RPE(pose_relation, delta=1, delta_unit=metrics.Unit.frames)
    rpe_metric.process_data(data)
    rpe_stat = rpe_metric.get_statistic(metrics.StatisticsType.rmse)
    rpe_stats = rpe_metric.get_all_statistics()

    Log("ATE - RMSE/Mean/Median/Std [m]", f"{ape_stats['rmse']:.4f}/{ape_stats['mean']:.4f}/{ape_stats['median']:.4f}/{ape_stats['std']:.4f}", tag="Eval")
    Log("ATE - Min/Max [m]", f"{ape_stats['min']:.4f}/{ape_stats['max']:.4f}", tag="Eval")
    Log("RPE - RMSE/Mean/Median/Std [m]", f"{rpe_stats['rmse']:.4f}/{rpe_stats['mean']:.4f}/{rpe_stats['median']:.4f}/{rpe_stats['std']:.4f}", tag="Eval")
    Log("RPE - Min/Max [m]", f"{rpe_stats['min']:.4f}/{rpe_stats['max']:.4f}", tag="Eval")

    with open(
            os.path.join(result_dir, "stats_{}.json".format(str(label))),
            "w",
            encoding="utf-8",
    ) as f:
        json.dump({"ape": ape_stats, "rpe": rpe_stats}, f, indent=4)

    plot_mode = evo.tools.plot.PlotMode.xy
    fig = plt.figure()
    ax = evo.tools.plot.prepare_axis(fig, plot_mode)
    ax.set_title(f"ATE RMSE: {ape_stat} RPE RMSE: {rpe_stat}")
    evo.tools.plot.traj(ax, plot_mode, traj_ref, "--", "gray", "gt")
    evo.tools.plot.traj_colormap(
        ax,
        traj_est_aligned,
        ape_metric.error,
        plot_mode,
        min_map=ape_stats["min"],
        max_map=ape_stats["max"],
    )
    ax.legend()
    plt.savefig(os.path.join(result_dir, "traj_{}.png".format(str(label))), dpi=90)

    return ape_stat, rpe_stat

def eval_ate(frames, kf_ids, save_dir, final=False, monocular=False):
    latest_frame_idx = kf_ids[-1] + 2 if final else kf_ids[-1] + 1

    all_trj_est_np, all_trj_gt_np = [], []
    kf_trj_est_np, kf_trj_gt_np = [], []

    def gen_pose_matrix(R, T):
        pose = np.eye(4)
        pose[0:3, 0:3] = R.cpu().numpy()
        pose[0:3, 3] = T.cpu().numpy()
        return pose

    for frame_idx in tqdm(range(len(frames)), desc="Processing all frames"):
        frame = frames[frame_idx]
        pose_est = np.linalg.inv(gen_pose_matrix(frame.R, frame.T))
        pose_gt = np.linalg.inv(gen_pose_matrix(frame.R_gt, frame.T_gt))
        all_trj_est_np.append(pose_est)
        all_trj_gt_np.append(pose_gt)

    for kf_id in tqdm(kf_ids, desc="Processing keyframes"):
        kf = frames[kf_id]
        pose_est = np.linalg.inv(gen_pose_matrix(kf.R, kf.T))
        pose_gt = np.linalg.inv(gen_pose_matrix(kf.R_gt, kf.T_gt))
        kf_trj_est_np.append(pose_est)
        kf_trj_gt_np.append(pose_gt)

    eval_type = "after_opt" if final else "before_opt"
    result_dir = os.path.join(save_dir, eval_type, "eval_slam")
    mkdir_p(result_dir)

    ate_all, rpe_all = evaluate_evo(
        poses_gt=all_trj_gt_np,
        poses_est=all_trj_est_np,
        result_dir=result_dir,
        label=f"allframes",
        monocular=monocular,
    )

    ate_kf, rpe_kf = evaluate_evo(
        poses_gt=kf_trj_gt_np,
        poses_est=kf_trj_est_np,
        result_dir=result_dir,
        label=f"keyframes",
        monocular=monocular,
    )

    ate_results = {
        "allframes_ate": float(ate_all),
        "allframes_rpe": float(rpe_all),
        "keyframes_ate": float(ate_kf),
        "keyframes_rpe": float(rpe_kf),
        "allframes_count": len(all_trj_est_np),
        "keyframes_count": len(kf_trj_est_np)
    }

    with open(os.path.join(result_dir, f"ate_results.json"), "w") as f:
        json.dump(ate_results, f, indent=4)

    wandb.log({"frame_idx": latest_frame_idx, "ate_all": ate_all, "ate_keyframes": ate_kf, "rpe_all": rpe_all, "rpe_keyframes": rpe_kf})
    return


def eval_rendering(frames, gaussians, dataset, save_dir, background, kf_indices, final=False):
    interval = 1
    end_idx = len(frames) - 1
    eval_type = "after_opt" if final else "before_opt"
    psnr_array, ssim_array, lpips_array = [], [], []
    train_psnr_array, train_ssim_array, train_lpips_array = [], [], []
    val_psnr_array, val_ssim_array, val_lpips_array = [], [], []
    cal_lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to("cuda")
    render_save_dir = os.path.join(save_dir, eval_type, "render")
    mkdir_p(render_save_dir)
    image_save_dir = os.path.join(save_dir, eval_type, "render/images")
    mkdir_p(image_save_dir)

    frame_results = []
    train_frame_results = []
    val_frame_results = []

    frame_indices = [idx for idx in range(0, end_idx, interval)]

    for idx in tqdm(frame_indices, desc="Evaluating frames"):
        frame = frames[idx]
        gt_image, _, _ = dataset[idx]

        rendering = render(frame, gaussians, background)["render"]
        image = torch.clamp(rendering, 0.0, 1.0)

        gt = (gt_image.cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        pred = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)

        render_filename = f"{idx:06d}.png"
        render_filepath = os.path.join(render_save_dir, render_filename)
        cv2.imwrite(render_filepath, pred)

        gt_filename = f"{idx:06d}.png"
        gt_filepath = os.path.join(image_save_dir, gt_filename)
        cv2.imwrite(gt_filepath, gt)

        mask = gt_image > 0

        psnr_score = psnr((image[mask]).unsqueeze(0), (gt_image[mask]).unsqueeze(0))
        ssim_score = ssim((image).unsqueeze(0), (gt_image).unsqueeze(0))
        lpips_score = cal_lpips((image).unsqueeze(0), (gt_image).unsqueeze(0))

        psnr_array.append(psnr_score.item())
        ssim_array.append(ssim_score.item())
        lpips_array.append(lpips_score.item())

        frame_result = {
            "frame_id": f"{idx:06d}",
            "psnr": psnr_score.item(),
            "ssim": ssim_score.item(),
            "lpips": lpips_score.item()
        }

        frame_results.append(frame_result)

        if idx in kf_indices:
            train_psnr_array.append(psnr_score.item())
            train_ssim_array.append(ssim_score.item())
            train_lpips_array.append(lpips_score.item())
            train_frame_results.append(frame_result)
        else:
            val_psnr_array.append(psnr_score.item())
            val_ssim_array.append(ssim_score.item())
            val_lpips_array.append(lpips_score.item())
            val_frame_results.append(frame_result)

    output = dict()
    output["mean_psnr"] = float(np.mean(psnr_array))
    output["mean_ssim"] = float(np.mean(ssim_array))
    output["mean_lpips"] = float(np.mean(lpips_array))
    output["frame_results"] = frame_results
    output["total_frames"] = len(frame_results)

    train_output = dict()
    train_output["mean_psnr"] = float(np.mean(train_psnr_array)) if train_psnr_array else 0.0
    train_output["mean_ssim"] = float(np.mean(train_ssim_array)) if train_ssim_array else 0.0
    train_output["mean_lpips"] = float(np.mean(train_lpips_array)) if train_lpips_array else 0.0
    train_output["frame_results"] = train_frame_results
    train_output["total_frames"] = len(train_frame_results)

    val_output = dict()
    val_output["mean_psnr"] = float(np.mean(val_psnr_array)) if val_psnr_array else 0.0
    val_output["mean_ssim"] = float(np.mean(val_ssim_array)) if val_ssim_array else 0.0
    val_output["mean_lpips"] = float(np.mean(val_lpips_array)) if val_lpips_array else 0.0
    val_output["frame_results"] = val_frame_results
    val_output["total_frames"] = len(val_frame_results)

    Log(
        f'mean psnr: {output["mean_psnr"]}, ssim: {output["mean_ssim"]}, lpips: {output["mean_lpips"]}',
        tag="Eval",
    )

    psnr_save_dir = os.path.join(save_dir, eval_type, "eval_render")
    mkdir_p(psnr_save_dir)

    json.dump(
        output,
        open(os.path.join(psnr_save_dir, "final_result.json"), "w", encoding="utf-8"),
        indent=4,
    )

    json.dump(
        train_output,
        open(os.path.join(psnr_save_dir, "train_result.json"), "w", encoding="utf-8"),
        indent=4,
    )

    json.dump(
        val_output,
        open(os.path.join(psnr_save_dir, "val_result.json"), "w", encoding="utf-8"),
        indent=4,
    )

    return output


def save_gaussians(gaussians, name, final=False):
    if name is None:
        return
    if final:
        point_cloud_path = os.path.join(name, "after_opt/point_cloud")
    else:
        point_cloud_path = os.path.join(name, "before_opt/point_cloud")
    mkdir_p(point_cloud_path)
    save_path = os.path.join(point_cloud_path, "point_cloud.ply")

    print(f"Saving Gaussian point cloud...")
    gaussians.save_ply(save_path)
