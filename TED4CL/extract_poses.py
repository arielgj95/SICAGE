import os
import cv2
import json
import time
import pickle
import math
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ----------------------------
# Scene detection (PySceneDetect)
# ----------------------------

def detect_scenes(video_path: str) -> List[Tuple[int, int]]:
    """Return scene (start_frame, end_frame) pairs using PySceneDetect content detector."""
    from scenedetect import VideoManager, SceneManager
    from scenedetect.detectors import ContentDetector

    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    base_timecode = video_manager.get_base_timecode()

    video_manager.set_downscale_factor(1)
    video_manager.start()

    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list(base_timecode)

    video_manager.release()
    return [(scene[0].get_frames(), scene[1].get_frames()) for scene in scene_list]


def extract_visible_scenes(
    video_path: str,
    output_folder: str,
    min_duration: float = 4.0,
    sample_fps: float = 15.0,
) -> None:
    """Save all the scenes lasting at least min_duration seconds in a JSON next to the video."""
    print(f"[Scenes] Processing video: {video_path}")

    os.makedirs(output_folder, exist_ok=True)

    scenes = detect_scenes(video_path)
    cap = cv2.VideoCapture(video_path)
    original_fps = float(cap.get(cv2.CAP_PROP_FPS))
    if original_fps <= 0:
        cap.release()
        raise RuntimeError(f"Could not read FPS from: {video_path}")

    frame_interval = max(1, int(round(original_fps / sample_fps)))
    scenes_info: List[Dict[str, Any]] = []

    for scene_start, scene_end in scenes:
        scene_duration = (scene_end - scene_start) / original_fps
        if scene_duration < min_duration:
            continue

        sampled_frames = list(range(scene_start, scene_end, frame_interval))
        scenes_info.append(
            {
                "start_frame": int(scene_start),
                "end_frame": int(scene_end),
                "duration_seconds": float(scene_duration),
                "video_fps": float(original_fps),
                "sample_fps_target": float(sample_fps),
                "sample_frame_interval": int(frame_interval),
                "sampled_frame_ids": sampled_frames,
            }
        )

    cap.release()

    scenes_output_path = os.path.splitext(video_path)[0] + "_scenes.json"
    with open(scenes_output_path, "w") as f:
        json.dump(scenes_info, f, indent=4)

    print(f"[Scenes] Saved: {scenes_output_path}")


# ----------------------------
# Dataset / folder utilities
# ----------------------------

def folder_has_required_modalities(video_folder_path: str) -> bool:
    """Require: >=1 mp4, >=1 mp3, and >=1 other file (subtitles or similar)."""
    mp3_count = 0
    mp4_count = 0
    other_files_count = 0

    for fn in os.listdir(video_folder_path):
        fp = os.path.join(video_folder_path, fn)
        if not os.path.isfile(fp):
            continue
        if fn.lower().endswith(".mp3"):
            mp3_count += 1
        elif fn.lower().endswith(".mp4"):
            mp4_count += 1
        else:
            other_files_count += 1

    return (mp3_count >= 1) and (mp4_count >= 1) and (other_files_count >= 1)


def find_video_name_in_folder(video_folder_path: str) -> Optional[str]:
    """Keep the original behavior: last .mp4 encountered wins if multiple exist."""
    video_name: Optional[str] = None
    for fn in os.listdir(video_folder_path):
        if fn.lower().endswith(".mp4"):
            video_name = fn
    return video_name


def outputs_exist_mmpose(video_path: str) -> Tuple[bool, bool, bool]:
    stem = os.path.splitext(video_path)[0]
    out3d = os.path.exists(stem + "_mmpose_data_output.pkl")
    out2d = os.path.exists(stem + "_mmpose_data_output_2d.pkl")
    outf = os.path.exists(stem + "_mmpose_data_output_final.pkl")
    return out3d, out2d, outf


# Compute video duration in seconds
def compute_video_duration(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    cap.release()
    return duration



# ----------------------------
# MMPose backend (local imports)
# ----------------------------

def build_mmpose_components(args) -> Dict[str, Any]:
    """
    Initialize detector, 2D estimator, 3D lifter, and (optional) visualizer.
    """
    try:
        import mmcv
        from mmpose.apis import (
            _track_by_iou,
            _track_by_oks,
            convert_keypoint_definition,
            extract_pose_sequence,
            inference_pose_lifter_model,
            inference_topdown,
            init_model,
        )
        from mmpose.models.pose_estimators import PoseLifter
        from mmpose.registry import VISUALIZERS
        from mmpose.structures import PoseDataSample, merge_data_samples, split_instances
        from mmpose.utils import adapt_mmdet_pipeline
        from mmdet.apis import inference_detector, init_detector
    except Exception as e:
        raise ImportError(
            "MMPose backend selected but required dependencies are missing "
            "(mmpose, mmcv, mmengine, mmdet)."
        ) from e

    detector = init_detector(args.det_config, args.det_checkpoint, device=args.device.lower())
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    pose_estimator = init_model(
        args.pose_estimator_config,
        args.pose_estimator_checkpoint,
        device=args.device.lower(),
    )

    pose_lifter = init_model(
        args.pose_lifter_config,
        args.pose_lifter_checkpoint if args.pose_lifter_checkpoint else None,
        device=args.device.lower(),
    )

    if not isinstance(pose_lifter, PoseLifter):
        raise TypeError('Only "PoseLifter" models are supported for stage-2 lifting.')

    det_kpt_color = pose_estimator.dataset_meta.get("keypoint_colors", None)
    det_dataset_skeleton = pose_estimator.dataset_meta.get("skeleton_links", None)
    det_dataset_link_color = pose_estimator.dataset_meta.get("skeleton_link_colors", None)

    pose_lifter.cfg.visualizer.radius = args.radius
    pose_lifter.cfg.visualizer.line_width = args.thickness
    pose_lifter.cfg.visualizer.det_kpt_color = det_kpt_color
    pose_lifter.cfg.visualizer.det_dataset_skeleton = det_dataset_skeleton
    pose_lifter.cfg.visualizer.det_dataset_link_color = det_dataset_link_color

    visualizer = None
    if not args.no_vis:
        visualizer = VISUALIZERS.build(pose_lifter.cfg.visualizer)
        visualizer.set_dataset_meta(pose_lifter.dataset_meta)

    return dict(
        mmcv=mmcv,
        inference_detector=inference_detector,
        inference_topdown=inference_topdown,
        inference_pose_lifter_model=inference_pose_lifter_model,
        extract_pose_sequence=extract_pose_sequence,
        convert_keypoint_definition=convert_keypoint_definition,
        _track_by_iou=_track_by_iou,
        _track_by_oks=_track_by_oks,
        PoseDataSample=PoseDataSample,
        merge_data_samples=merge_data_samples,
        split_instances=split_instances,
        detector=detector,
        pose_estimator=pose_estimator,
        pose_lifter=pose_lifter,
        visualizer=visualizer,
    )


# From mmpose code to map the keypoints from 2D coco to 3D.
# The mapping is used to compute detections scores in the same way
'''
# keypoints mapping coco -> h36m (from 2d to 3d)

if pose_lift_dataset in ['h36m', 'h3wb']:
    if pose_det_dataset in ['h36m', 'coco_wholebody']:
        keypoints_new = keypoints
    elif pose_det_dataset in ['coco', 'posetrack18']:
        
        # average the two visibilities to have the visibility of the central point 
        
        # pelvis (root) is in the middle of l_hip and r_hip
        keypoints_new[:, 0] = (keypoints[:, 11] + keypoints[:, 12]) / 2 
        # thorax is in the middle of l_shoulder and r_shoulder
        keypoints_new[:, 8] = (keypoints[:, 5] + keypoints[:, 6]) / 2
        
        # average the visibility of thorax and pelvis 
        # spine is in the middle of thorax and pelvis
        keypoints_new[:, 7] = (keypoints_new[:, 0] + keypoints_new[:, 8]) / 2
        
        # average the two visibilities to have the visibility of the central point 
        # in COCO, head is in the middle of l_eye and r_eye
        # in PoseTrack18, head is in the middle of head_bottom and head_top
        keypoints_new[:, 10] = (keypoints[:, 1] + keypoints[:, 2]) / 2
        # rearrange other keypoints
        keypoints_new[:, [1, 2, 3, 4, 5, 6, 9, 11, 12, 13, 14, 15, 16]] = \
            keypoints[:, [12, 14, 16, 11, 13, 15, 0, 5, 7, 9, 6, 8, 10]]
            
        #new keypoints to take for upper body
        [0, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] 
    
        #corresponding old keypoints - visibility scores (for upper body)
        [ (11 + 12) / 2, {(( 11 + 12) / 2) + ((5 + 6) / 2))} /2, (5 + 6) / 2, 0, ( 1 + 2) / 2, 5, 7, 9, 6, 8, 10] 
        
        #old to new all keypoints - visibility scores mapping
        [(11 + 12) / 2, 12, 14, 16, 11, 13, 15,  {(( 11 + 12) / 2) + ((5 + 6) / 2))}/2, (5 + 6) / 2 , 0, ( 1 + 2) / 2, 5, 7, 9, 6, 8, 10] 
        [0,             1,  2,  3,  4,  5,  6,    7,                                     8,           0, 10,           11,12,13,14,15,16]   
'''


# This code is similar to the official one from mmpose example code
def process_video_with_mmpose(
    video_path: str,
    output_folder: str,
    args,
    ctx: Dict[str, Any],
) -> None:

    """Visualize detected and predicted keypoints of one image.

    Pipeline of this function:

                              frame
                                |
                                V
                        +-----------------+
                        |     detector    |
                        +-----------------+
                                |  det_result
                                V
                        +-----------------+
                        |  pose_estimator |
                        +-----------------+
                                |  pose_est_results
                                V
            +--------------------------------------------+
            |  convert 2d kpts into pose-lifting format  |
            +--------------------------------------------+
                                |  pose_est_results_list
                                V
                    +-----------------------+
                    | extract_pose_sequence |
                    +-----------------------+
                                |  pose_seq_2d
                                V
                         +-------------+
                         | pose_lifter |
                         +-------------+
                                |  pose_lift_results
                                V
                       +-----------------+
                       | post-processing |
                       +-----------------+
                                |  pred_3d_data_samples
                                V
                         +------------+
                         | visualizer |
                         +------------+
    """

    os.makedirs(output_folder, exist_ok=True)

    mmcv = ctx["mmcv"]
    inference_detector = ctx["inference_detector"]
    inference_topdown = ctx["inference_topdown"]
    inference_pose_lifter_model = ctx["inference_pose_lifter_model"]
    extract_pose_sequence = ctx["extract_pose_sequence"]
    convert_keypoint_definition = ctx["convert_keypoint_definition"]
    _track_by_iou = ctx["_track_by_iou"]
    _track_by_oks = ctx["_track_by_oks"]
    PoseDataSample = ctx["PoseDataSample"]
    merge_data_samples = ctx["merge_data_samples"]
    split_instances = ctx["split_instances"]

    detector = ctx["detector"]
    pose_estimator = ctx["pose_estimator"]
    pose_lifter = ctx["pose_lifter"]
    visualizer = ctx["visualizer"]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Error opening video file {video_path}")

    video_fps = float(cap.get(cv2.CAP_PROP_FPS))
    if video_fps <= 0:
        cap.release()
        raise RuntimeError(f"Could not read FPS from: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    frame_skip_interval = max(1, int(round(video_fps / float(args.processing_fps))))
    pose_fps_effective = float(video_fps / frame_skip_interval)

    total_to_process = None
    if total_frames > 0:
        total_to_process = int(math.ceil(total_frames / frame_skip_interval))

    stem = os.path.splitext(video_path)[0]
    out_3d_pkl = stem + "_mmpose_data_output.pkl"
    out_2d_pkl = stem + "_mmpose_data_output_2d.pkl"
    out_vis_mp4 = stem + "_mmpose_output.mp4"

    if (not args.overwrite) and os.path.exists(out_3d_pkl) and os.path.exists(out_2d_pkl):
        print(f"[MMPose] Outputs exist; skipping extraction (overwrite={args.overwrite}): {video_path}")
        if not os.path.exists(stem + "_mmpose_data_output_final.pkl"):
            print("[MMPose] Final file missing; generating final from existing 2D/3D pickles.")
            add_detection_scores_mmpose(video_path, pose_fps=pose_fps_effective)
        cap.release()
        return

    if args.overwrite:
        for fp in [out_3d_pkl, out_2d_pkl, stem + "_mmpose_data_output_final.pkl", out_vis_mp4]:
            if os.path.exists(fp):
                try:
                    os.remove(fp)
                except OSError:
                    pass

    # Tracking / accumulation
    next_id = 0
    pose_est_results: List[Any] = []
    pose_est_results_list: List[List[Any]] = []  # list over time (processed frames)
    pred_instances_3d_list: List[Dict[str, Any]] = []
    pred_instances_2d_list: List[Dict[str, Any]] = []

    # Visualization video writer
    video_writer = None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    save_vis = (not args.no_save_video) and (visualizer is not None)

    # Loop state
    frame_idx_orig = -1
    processed_idx0 = -1  # 0-based index over processed frames

    log_interval = int(args.log_interval)
    max_frames = args.max_frames  # counts frames read

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_idx_orig += 1

            if max_frames is not None and frame_idx_orig >= max_frames:
                break

            if frame_idx_orig % frame_skip_interval != 0:
                continue

            processed_idx0 += 1
            frame_id_saved = processed_idx0 + 1  # keep your existing convention (first frame_id == 1)

            if log_interval > 0 and (frame_id_saved == 1 or frame_id_saved % log_interval == 0):
                if total_to_process is not None:
                    print(f"[MMPose] Processed {frame_id_saved}/{total_to_process} frames "
                          f"(orig frame {frame_idx_orig+1}/{total_frames if total_frames else '??'}).")
                else:
                    print(f"[MMPose] Processed {frame_id_saved} frames (orig frame {frame_idx_orig+1}).")

            pose_lift_dataset = pose_lifter.cfg.test_dataloader.dataset
            pose_lift_dataset_name = pose_lifter.dataset_meta["dataset_name"]

            pose_est_results_last = pose_est_results
            pose_est_results = []

            # Stage 1: detector
            det_result = inference_detector(detector, frame_bgr)
            pred_instance = det_result.pred_instances.cpu().numpy()

            bboxes = pred_instance.bboxes
            bboxes = bboxes[np.logical_and(
                pred_instance.labels == args.det_cat_id,
                pred_instance.scores > args.bbox_thr
            )]

            # Stage 1: top-down 2D pose
            pose_est_results = inference_topdown(pose_estimator, frame_bgr, bboxes)

            _track = _track_by_oks if args.use_oks_tracking else _track_by_iou
            pose_det_dataset_name = pose_estimator.dataset_meta["dataset_name"]

            # Convert 2D to lifter format + assign track ids
            pose_est_results_converted: List[Any] = []
            for i, data_sample in enumerate(pose_est_results):
                pred_instances = data_sample.pred_instances.cpu().numpy()
                keypoints = pred_instances.keypoints

                # areas/bboxes
                if "bboxes" in pred_instances:
                    areas = np.array([(bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                                      for bbox in pred_instances.bboxes])
                    pose_est_results[i].pred_instances.set_field(areas, "areas")
                else:
                    areas_list, bboxes_list = [], []
                    for keypoint in keypoints:
                        xmin = np.min(keypoint[:, 0][keypoint[:, 0] > 0], initial=1e10)
                        xmax = np.max(keypoint[:, 0])
                        ymin = np.min(keypoint[:, 1][keypoint[:, 1] > 0], initial=1e10)
                        ymax = np.max(keypoint[:, 1])
                        areas_list.append((xmax - xmin) * (ymax - ymin))
                        bboxes_list.append([xmin, ymin, xmax, ymax])
                    pose_est_results[i].pred_instances.areas = np.array(areas_list)
                    pose_est_results[i].pred_instances.bboxes = np.array(bboxes_list)

                # track id
                track_id, pose_est_results_last, _ = _track(
                    data_sample, pose_est_results_last, args.tracking_thr
                )

                if track_id == -1:
                    # If tracking fails, allow a new id only if enough visible keypoints
                    if np.count_nonzero(keypoints[:, :, 1]) >= int(args.min_kpts):
                        track_id = next_id
                        next_id += 1
                    else:
                        # delete instance
                        keypoints[:, :, 1] = -10
                        pose_est_results[i].pred_instances.set_field(keypoints, "keypoints")
                        pose_est_results[i].pred_instances.set_field(pred_instances.bboxes * 0, "bboxes")
                        pose_est_results[i].set_field(pred_instances, "pred_instances")
                        track_id = -1

                pose_est_results[i].set_field(track_id, "track_id")

                # convert keypoints order for pose lifting
                pose_est_result_converted = PoseDataSample()
                pose_est_result_converted.set_field(
                    pose_est_results[i].pred_instances.clone(), "pred_instances"
                )
                pose_est_result_converted.set_field(
                    pose_est_results[i].gt_instances.clone(), "gt_instances"
                )

                keypoints_converted = convert_keypoint_definition(
                    keypoints, pose_det_dataset_name, pose_lift_dataset_name
                )
                pose_est_result_converted.pred_instances.set_field(
                    keypoints_converted, "keypoints"
                )
                pose_est_result_converted.set_field(pose_est_results[i].track_id, "track_id")
                pose_est_results_converted.append(pose_est_result_converted)

            pose_est_results_list.append(pose_est_results_converted.copy())

            # Save 2D instances
            det_data_sample = merge_data_samples(pose_est_results)
            pred_2d_instances = det_data_sample.get("pred_instances", None)
            pred_instances_2d_list.append(
                dict(frame_id=int(frame_id_saved), instances=split_instances(pred_2d_instances))
            )

            # Stage 2: extract pose sequence + lift
            pose_seq_2d = extract_pose_sequence(
                pose_est_results_list,
                frame_idx=int(processed_idx0),  # 0-based index into pose_est_results_list
                causal=pose_lift_dataset.get("causal", False),
                seq_len=pose_lift_dataset.get("seq_len", 1),
                step=pose_lift_dataset.get("seq_step", 1),
            )

            if not pose_seq_2d:
                # no persons detected; still append empty frame to keep temporal alignment
                pred_instances_3d_list.append(dict(frame_id=int(frame_id_saved), instances=[]))
                continue

            norm_pose_2d = not args.disable_norm_pose_2d
            # WRONG! Before pass width, then height
            '''
            pose_lift_results = inference_pose_lifter_model(
                pose_lifter,
                pose_seq_2d,
                image_size=frame_bgr.shape[:2],
                norm_pose_2d=norm_pose_2d,
            )
            '''
            image_size = (frame_bgr.shape[1], frame_bgr.shape[0])  # (W, H)
            pose_lift_results = inference_pose_lifter_model(
                pose_lifter,
                pose_seq_2d,
                image_size=image_size,
                norm_pose_2d=norm_pose_2d,
            )

            # Postprocess
            for idx, pose_lift_result in enumerate(pose_lift_results):
                pose_lift_result.track_id = pose_est_results[idx].get("track_id", 1e4)

                pred_instances = pose_lift_result.pred_instances
                keypoints_3d = pred_instances.keypoints
                keypoint_scores = pred_instances.keypoint_scores

                if keypoint_scores is not None and keypoint_scores.ndim == 3:
                    keypoint_scores = np.squeeze(keypoint_scores, axis=1)
                    pose_lift_results[idx].pred_instances.keypoint_scores = keypoint_scores

                if keypoints_3d.ndim == 4:
                    keypoints_3d = np.squeeze(keypoints_3d, axis=1)

                # swap axes and flip signs as in demo
                keypoints_3d = keypoints_3d[..., [0, 2, 1]]
                keypoints_3d[..., 0] = -keypoints_3d[..., 0]
                keypoints_3d[..., 2] = -keypoints_3d[..., 2]

                if not args.disable_rebase_keypoint:
                    keypoints_3d[..., 2] -= np.min(keypoints_3d[..., 2], axis=-1, keepdims=True)

                pose_lift_results[idx].pred_instances.keypoints = keypoints_3d

            # Keep official-like ordering by track_id for stability
            pose_lift_results = sorted(pose_lift_results, key=lambda x: x.get("track_id", 1e4))

            pred_3d_data_samples = merge_data_samples(pose_lift_results)
            pred_3d_instances = pred_3d_data_samples.get("pred_instances", None)

            # Visualization
            if save_vis:
                visualize_frame = mmcv.bgr2rgb(frame_bgr)
                visualizer.add_datasample(
                    "result",
                    visualize_frame,
                    data_sample=pred_3d_data_samples,
                    det_data_sample=det_data_sample,
                    draw_gt=False,
                    dataset_2d=pose_det_dataset_name,
                    dataset_3d=pose_lift_dataset_name,
                    show=args.show,
                    draw_bbox=True,
                    kpt_thr=args.kpt_thr,
                    num_instances=args.num_instances,
                    wait_time=args.show_interval,
                )
                frame_vis_rgb = visualizer.get_image()
                frame_vis_bgr = mmcv.rgb2bgr(frame_vis_rgb)

                if video_writer is None:
                    video_writer = cv2.VideoWriter(
                        out_vis_mp4,
                        fourcc,
                        pose_fps_effective,
                        (frame_vis_bgr.shape[1], frame_vis_bgr.shape[0]),
                    )
                video_writer.write(frame_vis_bgr)

            # Save 3D instances
            pred_instances_3d_list.append(
                dict(frame_id=int(frame_id_saved), instances=split_instances(pred_3d_instances))
            )

            if args.show:
                if cv2.waitKey(5) & 0xFF == 27:
                    break
                time.sleep(float(args.show_interval))

    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()

    # Save 3D pickle
    with open(out_3d_pkl, "wb") as f:
        pickle.dump(
            dict(meta_info=pose_lifter.dataset_meta, instance_info=pred_instances_3d_list),
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(f"[MMPose] Saved: {out_3d_pkl}")

    # Save 2D pickle
    with open(out_2d_pkl, "wb") as f:
        pickle.dump(
            dict(meta_info=pose_lifter.dataset_meta, instance_info=pred_instances_2d_list),
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(f"[MMPose] Saved: {out_2d_pkl}")

    # Build final pickle with fps + pose_fps (
    add_detection_scores_mmpose(video_path, pose_fps=pose_fps_effective)
    print(f"[MMPose] Done: {video_path}")


def add_detection_scores_mmpose(video_path, pose_fps=15.):
    """This function is used in case 2D confidence detection scores were not added to 3D poses
    (the pose lifter doesn't have confidence values).  Also, scene fps and pose fps are added.
    These are necessary for correct pose processing"""

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    stem = os.path.splitext(video_path)[0]
    poses_3d_data_path = stem + "_mmpose_data_output.pkl"
    poses_2d_data_path = stem + "_mmpose_data_output_2d.pkl"
    data_save_path_final = stem + "_mmpose_data_output_final.pkl"

    if os.path.exists(poses_2d_data_path):
        with open(poses_2d_data_path, "rb") as f:
            poses_data_2d = pickle.load(f)
            #print(poses_data_2d['instance_info'][0]['instances'])
            pass
    else:
        raise FileNotFoundError(f"Pickle file 2d doesn't exists!!")

    if os.path.exists(poses_3d_data_path):
        with open(poses_3d_data_path, "rb") as f:
            poses_data_3d = pickle.load(f)
            #print(poses_data_3d['instance_info'][0]['instances'])
            pass
    else:
        raise FileNotFoundError(f"Pickle file 3d doesn't exists!!")

    '''
    if os.path.exists(data_save_path_final):
        with open(data_save_path_final, "rb") as f:
            poses_data_final = pickle.load(f)
            #print(poses_data_3d['instance_info'][0]['instances'])
            pass
    else:
        raise FileNotFoundError(f"Pickle file doesn't exists!!")
    '''

    poses_data_final: Dict[str, Any] = {}
    poses_data_final["meta_info"] = poses_data_3d["meta_info"].copy()
    poses_data_final["instance_info"] = []

    poses_data_final["meta_info"]["fps"] = float(fps)
    poses_data_final["meta_info"]["pose_fps"] = float(pose_fps)

    n = min(len(poses_data_2d["instance_info"]), len(poses_data_3d["instance_info"]))

    for i in range(n):
        keypoint_set_2d = poses_data_2d["instance_info"][i]
        keypoint_set_3d = poses_data_3d["instance_info"][i]

        frame_info = {"frame_id": keypoint_set_2d["frame_id"], "instances": []}
        inst2d_list = keypoint_set_2d.get("instances", [])
        inst3d_list = keypoint_set_3d.get("instances", [])
        #print("LEN",len(keypoint_set_2d['instances']))
        m = min(len(keypoint_set_2d["instances"]), len(keypoint_set_3d["instances"]))
        for j in range(m):
            person_2d = inst2d_list[j]
            person_3d = inst3d_list[j]

            instance: Dict[str, Any] = {}
            instance['keypoints'] = person_3d['keypoints']
            #coco to h36m keypoints - visibility scores mapping
            #        #old to new all keypoints - visibility scores mapping
            #[(11 + 12) / 2, 12, 14, 16, 11, 13, 15,  {(( 11 + 12) / 2) + ((5 + 6) / 2))}/2, (5 + 6) / 2 , 0, ( 1 + 2) / 2, 5, 7, 9, 6, 8, 10]
            #[0,             1,  2,  3,  4,  5,  6,    7,                                     8,           0, 10,           11,12,13,14,15,16]

            keypoint_scores = person_2d['keypoint_scores']
            #print(keypoint_scores)
            instance['keypoint_scores'] = [
                (keypoint_scores[11] + keypoint_scores[12]) / 2,  # Average of keypoints 11 and 12
                keypoint_scores[12],  # Keypoint 12
                keypoint_scores[14],  # Keypoint 14
                keypoint_scores[16],  # Keypoint 16
                keypoint_scores[11],  # Keypoint 11
                keypoint_scores[13],  # Keypoint 13
                keypoint_scores[15],  # Keypoint 15
                ((keypoint_scores[11] + keypoint_scores[12]) / 2 + (keypoint_scores[5] + keypoint_scores[6]) / 2) / 2,
                # Average of midpoints of (11,12) and (5,6)
                (keypoint_scores[5] + keypoint_scores[6]) / 2,  # Average of keypoints 5 and 6
                keypoint_scores[0],  # Keypoint 0
                (keypoint_scores[1] + keypoint_scores[2]) / 2,  # Average of keypoints 1 and 2
                keypoint_scores[5],  # Keypoint 5
                keypoint_scores[7],  # Keypoint 7
                keypoint_scores[9],  # Keypoint 9
                keypoint_scores[6],  # Keypoint 6
                keypoint_scores[8],  # Keypoint 8
                keypoint_scores[10]  # Keypoint 10
            ]
            #print(instance['keypoint_scores'])
            #instance['keypoint_scores'] = person_2d['keypoint_scores']
            instance['bbox'] = person_2d['bbox']
            instance['bbox_score'] = person_2d['bbox_score']
            frame_info['instances'].append(instance)
        poses_data_final['instance_info'].append(frame_info)

    with open(data_save_path_final, 'wb') as f:
        pickle.dump(poses_data_final,f,protocol=pickle.HIGHEST_PROTOCOL)
        print(f"SAVED {data_save_path_final}")

        #print(poses_data_final['instance_info'][i],"\n",poses_data_2d['instance_info'][i])


# ----------------------------
# MediaPipe backend (local imports)
# ----------------------------

def process_video_with_mediapipe(video_path: str, output_folder: str, args) -> None:
    """
    MediaPipe processing. Imports are local.
    Saves: <stem>_mediapipe_data_output.pkl and optionally a vis video.
    """
    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        from mediapipe.framework.formats import landmark_pb2
        from mediapipe import solutions
    except Exception as e:
        raise ImportError("MediaPipe backend selected but mediapipe is not available.") from e

    os.makedirs(output_folder, exist_ok=True)
    stem = os.path.splitext(video_path)[0]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"[ERROR] Could not open video: {video_path}")

    video_fps = float(cap.get(cv2.CAP_PROP_FPS))
    if video_fps <= 0:
        cap.release()
        raise RuntimeError(f"Could not read FPS from: {video_path}")

    frame_skip_interval = max(1, int(round(video_fps / float(args.processing_fps))))
    pose_fps_effective = float(video_fps / frame_skip_interval)

    base_options = python.BaseOptions(model_asset_path=str(args.mediapipe_model_path))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        output_segmentation_masks=True,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    video_writer = None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video_path = f"{stem}_mediapipe_output.mp4"

    def draw_landmarks(rgb_image: np.ndarray, detection_result) -> np.ndarray:
        annotated_image = np.copy(rgb_image)
        for pose_landmarks in detection_result.pose_landmarks:
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend(
                [landmark_pb2.NormalizedLandmark(x=l.x, y=l.y, z=l.z) for l in pose_landmarks]
            )
            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                pose_landmarks_proto,
                solutions.pose.POSE_CONNECTIONS,
                solutions.drawing_styles.get_default_pose_landmarks_style(),
            )
        return annotated_image

    frames_out: List[Dict[str, Any]] = []
    frame_idx_orig = -1
    processed_idx0 = -1

    log_interval = int(args.log_interval)
    max_frames = args.max_frames

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_idx_orig += 1

            if max_frames is not None and frame_idx_orig >= max_frames:
                break

            if frame_idx_orig % frame_skip_interval != 0:
                continue

            processed_idx0 += 1
            frame_id_saved = processed_idx0 + 1

            if log_interval > 0 and (frame_id_saved == 1 or frame_id_saved % log_interval == 0):
                print(f"[MediaPipe] Processed {frame_id_saved} frames (orig frame {frame_idx_orig+1}).")

            timestamp_ms = int((frame_idx_orig / video_fps) * 1000)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
            )
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            poses_serialized = []
            for pose_landmarks in result.pose_landmarks:
                poses_serialized.append(
                    [
                        {
                            "x": float(l.x),
                            "y": float(l.y),
                            "z": float(l.z),
                            "visibility": float(getattr(l, "visibility", 0.0)),
                        }
                        for l in pose_landmarks
                    ]
                )

            frames_out.append(
                dict(
                    frame_id=int(frame_id_saved),
                    orig_frame_id=int(frame_idx_orig),
                    timestamp_sec=float(frame_idx_orig / video_fps),
                    poses=poses_serialized,
                )
            )

            if not args.no_save_video:
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                annotated = draw_landmarks(rgb, result)
                annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

                if video_writer is None:
                    video_writer = cv2.VideoWriter(
                        out_video_path,
                        fourcc,
                        pose_fps_effective,
                        (annotated_bgr.shape[1], annotated_bgr.shape[0]),
                    )
                video_writer.write(annotated_bgr)

    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()
        landmarker.close()

    payload = dict(
        meta_info=dict(
            video_path=video_path,
            fps=float(video_fps),
            processing_fps_target=float(args.processing_fps),
            frame_skip_interval=int(frame_skip_interval),
            pose_fps=float(pose_fps_effective),
            backend="mediapipe",
            mediapipe_model_path=str(args.mediapipe_model_path),
        ),
        instance_info=frames_out,
    )

    out_pkl = f"{stem}_mediapipe_data_output.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[MediaPipe] Saved: {out_pkl}")
    print(f"[MediaPipe] Done: {video_path}")


def process_single_video(video_path: str, args, mmp_ctx: Optional[Dict[str, Any]]) -> None:
    video_path = os.path.abspath(video_path)
    out_dir = os.path.dirname(video_path)

    if args.enable_scenes:
        scenes_json = os.path.splitext(video_path)[0] + "_scenes.json"
        if args.overwrite_scenes or (not os.path.exists(scenes_json)):
            extract_visible_scenes(
                video_path,
                out_dir,
                min_duration=float(args.scene_min_duration),
                sample_fps=float(args.scene_sample_fps),
            )

    if args.backend == "mmpose":
        assert mmp_ctx is not None
        process_video_with_mmpose(video_path, out_dir, args, mmp_ctx)
    else:
        process_video_with_mediapipe(video_path, out_dir, args)

# Extract all the poses from playlists
def process_playlists(playlists_folder: str, args, mmp_ctx: Optional[Dict[str, Any]]) -> None:
    video_counter = 0
    playlists_folder = os.path.abspath(playlists_folder)

    for playlist in os.listdir(playlists_folder):
        playlist_path = os.path.join(playlists_folder, playlist)
        if not os.path.isdir(playlist_path):
            continue

        for video_folder in os.listdir(playlist_path):
            video_folder_path = os.path.join(playlist_path, video_folder)
            if not os.path.isdir(video_folder_path):
                continue

            video_counter += 1

            if not folder_has_required_modalities(video_folder_path):
                print(f"[Skip] {video_folder_path} must contain >=1 .mp3, >=1 .mp4, and >=1 other file.")
                continue

            video_name = find_video_name_in_folder(video_folder_path)
            if video_name is None:
                print(f"[Skip] No .mp4 found in {video_folder_path}")
                continue

            video_path = os.path.join(video_folder_path, video_name)

            # Scenes
            if args.enable_scenes:
                scenes_json = os.path.splitext(video_path)[0] + "_scenes.json"
                if args.overwrite_scenes or (not os.path.exists(scenes_json)):
                    extract_visible_scenes(
                        video_path,
                        video_folder_path,
                        min_duration=float(args.scene_min_duration),
                        sample_fps=float(args.scene_sample_fps),
                    )

            # Backend
            if args.backend == "mmpose":
                out3d, out2d, outf = outputs_exist_mmpose(video_path)

                if out3d and out2d and outf and (not args.overwrite):
                    print(f"[Skip] Already processed: {video_path}")
                    continue

                if out3d and out2d and (not outf) and (not args.overwrite):
                    print(f"[Final] Creating final poses file for: {video_path}")
                    # We need correct pose_fps; recompute from current args and video fps.
                    cap = cv2.VideoCapture(video_path)
                    fps = float(cap.get(cv2.CAP_PROP_FPS))
                    cap.release()
                    frame_skip_interval = max(1, int(round(fps / float(args.processing_fps))))
                    pose_fps_effective = float(fps / frame_skip_interval)
                    add_detection_scores_mmpose(video_path, pose_fps=pose_fps_effective)
                    continue

                print(f"[MMPose] Processing video ({video_counter}): {video_path}")
                assert mmp_ctx is not None
                process_video_with_mmpose(video_path, video_folder_path, args, mmp_ctx)

            else:
                print(f"[MediaPipe] Processing video ({video_counter}): {video_path}")
                process_video_with_mediapipe(video_path, video_folder_path, args)

# ----------------------------
# Argparse
# ----------------------------

def build_arg_parser() -> ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    p = ArgumentParser(description="Two-stage 3D pose extraction (MMPose) or MediaPipe pose extraction.")

    p.add_argument("--playlists-folder", type=str, default="",
                   help="Root folder containing playlists.")
    p.add_argument("--video-path", type=str, default="",
                   help="If set, process only this single video instead of playlists-folder tree.")

    p.add_argument("--backend", choices=["mmpose", "mediapipe"], default="mmpose")
    p.add_argument("--processing-fps", type=float, default=15.0,
                   help="Target processing fps (frames are skipped to approximate this).")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Optional cap on number of ORIGINAL frames read (debug). Default: full video.")
    p.add_argument("--overwrite", action="store_true", default=False,
                   help="Overwrite existing outputs.")
    p.add_argument("--log-interval", type=int, default=1000,
                   help="Print progress every N processed frames (0 disables).")

    # Scenes: default enabled, but allow disabling
    g = p.add_mutually_exclusive_group()
    g.add_argument("--enable-scenes", dest="enable_scenes", action="store_true", default=True)
    g.add_argument("--disable-scenes", dest="enable_scenes", action="store_false")

    p.add_argument("--overwrite-scenes", action="store_true", default=False)
    p.add_argument("--scene-min-duration", type=float, default=4.0)
    p.add_argument("--scene-sample-fps", type=float, default=15.0)

    # MMPose configs/checkpoints
    p.add_argument("--det-config", type=str,
                   default=str(repo_root / "mmpose/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py"))
    p.add_argument("--det-checkpoint", type=str,
                   default="https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth")#https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"
    p.add_argument("--pose-estimator-config", type=str,
                   default=str(repo_root / "mmpose/configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb256-420e_body8-256x192.py"))

    p.add_argument("--pose-estimator-checkpoint", type=str,
                   default="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth") #https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth"

    p.add_argument("--pose-lifter-config", type=str,
                   default=str(repo_root / "mmpose/configs/body_3d_keypoint/motionbert/h36m/"
                               "motionbert_dstformer-ft-243frm_8xb32-120e_h36m.py"))
    p.add_argument("--pose-lifter-checkpoint", type=str,
                   default=str(repo_root / "mmpose/motionbert_ft_h36m-d80af323_20230531.pth"))

    # Device / thresholds / tracking
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--det-cat-id", type=int, default=0)
    p.add_argument("--bbox-thr", type=float, default=0.3)
    p.add_argument("--kpt-thr", type=float, default=0.3)
    p.add_argument("--use-oks-tracking", action="store_true", default=False)
    p.add_argument("--tracking-thr", type=float, default=0.3)
    p.add_argument("--min-kpts", type=int, default=3)

    p.add_argument("--show", action="store_true", default=False)
    p.add_argument("--show-interval", type=float, default=0.0)
    p.add_argument("--thickness", type=int, default=1)
    p.add_argument("--radius", type=int, default=3)
    p.add_argument("--num-instances", type=int, default=1)

    p.add_argument("--disable-rebase-keypoint", action="store_true", default=False)
    p.add_argument("--disable-norm-pose-2d", action="store_true", default=False)

    # Output toggles
    p.add_argument("--no-save-video", action="store_true", default=False)
    p.add_argument("--no-vis", action="store_true", default=False)

    # MediaPipe args
    p.add_argument(
        "--mediapipe-model-path",
        type=str,
        default=str(repo_root / "pose_landmarker_lite.task"),
    )

    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    if not args.video_path and not args.playlists_folder:
        raise ValueError("You must set either --video-path or --playlists-folder.")

    mmp_ctx = None
    if args.backend == "mmpose":
        mmp_ctx = build_mmpose_components(args)

    if args.video_path:
        process_single_video(args.video_path, args, mmp_ctx)
    else:
        process_playlists(args.playlists_folder, args, mmp_ctx)


if __name__ == "__main__":
    main()
