import os
import json
import math
import pickle
from argparse import ArgumentParser
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R

import time
#pose_fps = 15  #poses were extracted at 15 fps

'''
h36m keypoints:
0: hip  #this keypoint is wrongly estimated when passing from 2D data to 3D. Better to use number 7
1: right-hip
2: right-knee
3: right-foot
4: left-hip
5: left-knee
6: left-foot
7: spine-centre
8: spine-upper #note that neck in this case not considered as the middle between shoulders but as first part of head
9: neck 
10:head
11:left-shoulder
12:left-elbow
13:left-wrist
14:right-shoulder
15:right-elbow
16:right-wrist
'''


"""
This script processes previously generated 3D poses (e.g., MMPose pickles) into clean motion data.
It expects pose pickles with the structure:

data.keys() -> dict_keys(['meta_info', 'instance_info'])

data['instance_info'] is a list of frames, each like:
  {'frame_id': int, 'instances': [ { 'keypoints': ..., 'keypoint_scores': ..., (optional bbox fields) }, ... ] }

Scenes are loaded from a JSON created by your scene extractor (PySceneDetect), with items like:
  {'start_frame': int, 'end_frame': int, ...}

The main output is stored in:
  <video_folder>/all_motion_data/motion_<init_start>_<init_end>.pkl 
   where <init_start>_<init_end> are respectively the start and the end frame of the motion sequence in the video

and contains at minimum:
  dict(init_start=..., init_end=..., real_start=..., real_end=..., data=<6D rotations>)
"""


class MotionPreprocessor:

    def __init__(
        self,
        poses: List[Dict[str, Any]],
        scene: Tuple[int, int],
        scene_fps: float,
        keypoint_indices: List[int],
        visibility_th: float = 0.2,
        presence_th: float = 0.9,
        min_valid_keypoints: int = 5,
        pose_fps: float = 15.0,
        min_duration_sec: float = 4.0,
        skeleton_info: Dict[str, Any] = None,
        save_path: str = "",
        pose_start_frame_orig: Optional[int] = None,
        enable_debug_gifs: bool = True,
        enable_keypoint_fixes: bool = True,
    ):
        self.poses = poses
        self.initial_start = int(scene[0])
        self.initial_end = int(scene[1])
        self.scene_fps = float(scene_fps)

        self.keypoint_indices = keypoint_indices
        self.visibility_th = float(visibility_th)
        self.presence_th = float(presence_th)
        self.min_valid_keypoints = int(min_valid_keypoints)
        self.pose_fps = float(pose_fps)
        self.min_duration_sec = float(min_duration_sec)

        self.skeleton_info = skeleton_info or {}
        self.save_path = save_path

        # If pose_start_frame_orig is provided, it is the original video-frame index
        # corresponding to the first pose frame in self.poses.
        # This helps compute real_start/real_end correctly even when the scene start
        # is not aligned to the pose sampling grid.
        self.pose_start_frame_orig = int(pose_start_frame_orig) if pose_start_frame_orig is not None else self.initial_start

        self.main_speaker_skeletons: List[Dict[str, Any]] = []
        self.parent_indices: List[int] = []
        self.avg_segment_lengths: Optional[np.ndarray] = None

        self.downsample_factor = max(1, int(round(self.scene_fps / self.pose_fps)))
        self.enable_debug_gifs = bool(enable_debug_gifs)
        self.enable_keypoint_fixes = bool(enable_keypoint_fixes)

    # ----------------------------
    # Top-level scene processing
    # ----------------------------

    def process_scene_motion(self) -> Tuple[Optional[bool], str]:
        #Here we save all the available poses of the main speaker in the video. Each scene is saved as motion_scene_start_end.pkl
        all_motions_save_path = os.path.join(self.save_path,"all_motion_data")
        _ensure_dir(all_motions_save_path)

        sample_save_path = os.path.join(
            all_motions_save_path,
            f"motion_{self.initial_start}_{self.initial_end}.pkl"
        )

        if os.path.exists(sample_save_path):
            print(f"[Skip] motion_{self.initial_start}_{self.initial_end} already processed.")
            return None, "sample already processed"


        # If there are more than 4 speakers for most of the video, do not process it
        if self.is_many_people():
            return None, "there are too many people"  # Discard scenes with too many people
        print(f"[Scene] Processing START={self.initial_start}, END={self.initial_end}")

        # Extract the main speaker's skeleton over time
        self.main_speaker_skeletons = self.find_main_speaker_skeletons()

        #for skeleton in self.main_speaker_skeletons:
        #    print("Main Skeleton:",skeleton)
        if len(self.main_speaker_skeletons) == 0: #No main speaker keypoints in the scene
            return None, "there is no main speaker data"

        if not self.keypoints_presence(): # keypoints are not sufficiently present
            return None, "there aren't enough keypoints"  # Discard this window


        # Fill missing keypoints (trims to a globally-valid interval)
        data , start_frame, end_frame = self.interpolate_keypoints()

        if not data:
            return None, "can't perform the interpolation"
        else:
            self.main_speaker_skeletons = data


        # Create parent indices from skeleton metadata.
        self.create_parent_indices()

        # Smooth the motion with Savgol filter to make motion less shaky
        self.smooth_motion()

        if self.enable_debug_gifs:
            self.plot_poses_animation(
                all_motions_save_path,
                f"motion_{self.initial_start}_{self.initial_end}",
                len(self.main_speaker_skeletons),
            )


        # Align hip to origin [0,0,0].
        if 7 in self.keypoint_indices:
            hip_idx = self.keypoint_indices.index(7)
        else:
            hip_idx = 0  # fallback: first selected joint
        self.align_hip_with_global_frame(hip_index=hip_idx)

        # Remove the average global orientation around Z using spine + shoulders.
        # Indices are looked up in self.keypoint_indices so subsets are supported.
        self.remove_orientation_around_z(shoulder1_ind = self.keypoint_indices.index(11), #11
                                         shoulder2_ind = self.keypoint_indices.index(14),
                                         spine1_ind = self.keypoint_indices.index(7),
                                         spine2_ind = self.keypoint_indices.index(8),
                                         head_ind = self.keypoint_indices.index(9)) #14

        # Debug visualization (kept; optional via CLI).
        if self.enable_debug_gifs:
            self.plot_poses_animation(
                all_motions_save_path,
                f"motion_final_{self.initial_start}_{self.initial_end}",
                len(self.main_speaker_skeletons),
            )


        # Filter out sideways/back-facing skeletons.
        if self.is_skeleton_sideways() or self.is_skeleton_back():
            return None, "keypoint aren't clearly visible"


        # Compute average link length for motion reconstruction and save.
        self.compute_person_avg_segment_length()


        # Compute rotation matrices between parents and children.
        all_rotation_matrices = self.compute_rotation_matrices()

        # Compute relative rotations between parents and children.
        relative_rotations = self.represent_relative_rotations(all_rotation_matrices)

        # Find 6D representation.
        sixd_representations = self.np_matrix_to_rotation_6d(relative_rotations)


        '''
        ######## REVERSE THE PROCESS FOR DEBUGGING
        reversed_relative_matrices = self.rotation_6d_to_matrix(torch.from_numpy(sixd_representations))
        reversed_matrices = self.convert_to_original_representation(reversed_relative_matrices)
        reversed_motion = self.compute_joint_positions(reversed_matrices, self.avg_segment_lengths)

        self.main_speaker_skeletons = reversed_motion # Assign reversed_motion to elf.main_speaker_skeletons to plot

        if self.enable_debug_gifs:
            self.plot_poses_animation(
                all_motions_save_path,
                f"motion_reversed_{self.initial_start}_{self.initial_end}",
                len(self.main_speaker_skeletons),
            )
        '''
        # Save data
        payload = dict(
            init_start=self.initial_start,
            init_end=self.initial_end,
            real_start=int(start_frame),
            real_end=int(end_frame),
            data=sixd_representations,
            # Extra debugging info is safe to add (downstream scripts can ignore it).
            pose_start_frame_orig=int(self.pose_start_frame_orig),
            pose_fps=float(self.pose_fps),
            scene_fps=float(self.scene_fps),
        )
        with open(sample_save_path, 'wb') as f:  # save links length
            pickle.dump(dict(init_start = self.initial_start,
                             init_end = self.initial_end,
                             real_start = start_frame,
                             real_end = end_frame,
                             data = sixd_representations),
                             f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(sample_save_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

        return True, "everything is fine"

    # ----------------------------
    # Motion reconstruction from 6D representation
    # ----------------------------

    def convert_to_original_representation(self, relative_rotations):
        """
        Converts relative rotation matrices back to the original global rotation representation.
        """
        n_frames, n_joints, _, _ = relative_rotations.shape
        original_representation = np.zeros_like(relative_rotations)

        # Initialize the first frame
        original_representation[0] = relative_rotations[0]

        # Apply relative rotations sequentially
        for f in range(1, n_frames):
            for j in range(n_joints):
                parent_index = self.parent_indices[j]
                if parent_index >= 0:
                    parent_rotation = original_representation[f - 1, parent_index]
                else:
                    parent_rotation = np.identity(3)
                original_representation[f, j] = np.dot(relative_rotations[f, j], parent_rotation)

        return original_representation


    def compute_joint_positions(self, rotation_matrices, segment_lengths):
        """
        Computes the global positions of each joint based on rotation matrices and segment lengths.
        """
        #if self.avg_segment_lengths is None and segment_lengths is None:
            #self.compute_person_avg_segment_length()
            #raise ValueError("Average segment lengths not computed. Call compute_person_avg_segment_length() first.")

        n_frames, n_joints, _, _ = rotation_matrices.shape
        joint_positions = np.zeros((n_frames, n_joints, 3))
        segment_lengths = segment_lengths.squeeze()  # Ensure shape is (n_joints,)

        for frame in range(n_frames):
            for joint in range(n_joints):
                parent_index = self.parent_indices[joint]
                segment_length = segment_lengths[joint]
                rotation_matrix = rotation_matrices[frame, joint]

                if parent_index >= 0:
                    parent_position = joint_positions[frame, parent_index]
                    offset = np.dot(rotation_matrix, np.array([0, 0, segment_length]))
                    joint_positions[frame, joint, :] = parent_position + offset
                else:
                    # For root joint, position is determined solely by the offset
                    offset = np.dot(rotation_matrix, np.array([0, 0, segment_length]))
                    joint_positions[frame, joint, :] = offset

        return joint_positions



    def compute_joint_positions_from_rotations(self, rotation_matrices, segment_lengths=None):
        """
        Computes global joint positions from rotation matrices and segment lengths.
        """
        if segment_lengths is None:
            if self.avg_segment_lengths is None:
                self.compute_person_avg_segment_length()
            segment_lengths = self.avg_segment_lengths
        joint_positions = self.compute_joint_positions(rotation_matrices, segment_lengths)
        return joint_positions


    # ----------------------------
    # Speaker selection and tracking
    # ----------------------------

    def is_many_people(self) -> bool:
        n_people = [len(p.get("instances", [])) for p in self.poses]
        return len(n_people) > 0 and float(np.mean(n_people)) > 4.0

    def _h36m_to_local(self) -> Dict[int, int]:
        """Map H36M id -> local index within self.keypoint_indices."""
        return {h36m_id: i for i, h36m_id in enumerate(self.keypoint_indices)}

    def _idx_or_raise(self, h36m_id: int) -> int:
        m = self._h36m_to_local()
        if h36m_id not in m:
            raise ValueError(f"Required keypoint {h36m_id} not present in keypoint_indices={self.keypoint_indices}")
        return m[h36m_id]

    def get_skeleton_from_frame(self, instance: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract the skeleton data from an instance based on selected keypoint indices.

        Note:
        - In the "non-original" mmpose configuration of humanbert, some keypoints do look wrongly estimated.
          In such case, manual corrections on keypoints 0, 9 and 10 are necessary. We decide whether or not
          applying this fix depending by putting to True the value of the self.enable_keypoint_fixes flag.
          This prevents "fixing" already good data.
        """
        keypoints = instance.get("keypoints", [])
        keypoint_scores = instance.get("keypoint_scores", [])

        # Pull the raw selected keypoints first.
        raw: Dict[int, List[float]] = {}
        raw_scores: Dict[int, float] = {}
        for h36m_id in self.keypoint_indices:
            if h36m_id < len(keypoints):
                raw[h36m_id] = list(keypoints[h36m_id])
                raw_scores[h36m_id] = float(keypoint_scores[h36m_id])
            else:
                # Should not happen if the pipeline inserts zeros for missing keypoints, but we keep a safe fallback.
                raw[h36m_id] = [0.0, 0.0, 0.0]
                raw_scores[h36m_id] = 0.0
        # Extract joints used by the heuristic fixes (if present).
        # kp 0 is central hip; kp 1 and 4 are the left and right hips.
        rhip = raw.get(1, None)
        lhip = raw.get(4, None)
        kp0 = raw.get(0, None)

        #  8 is lower neck reference, and kp 9/10 as head/forehead-like joints.
        ref_neck_like = raw.get(8, None)
        head_like = raw.get(9, None)
        forehead_like = raw.get(10, None)

        # Compute a rough body scale for misalignment tests.
        # Shoulder width is a stable scale when shoulders are visible.
        scale = self._estimate_body_scale(raw, raw_scores)

        # ---------------------------
        # Manual fix: kp 0
        # ---------------------------
        # Adjustments to make the predicted keypoints more consistent w.r.t original keypoints.
        # These are made heuristically as some predicted keypoints are inconsistent with real ones.
        # These are specifically applied for the "non-original" configuration of humanbert during
        # poses extraction with MMPose.
        #
        # We apply this fix only if kp0 is clearly misaligned w.r.t the midpoint of left/right hip.
        if self.enable_keypoint_fixes and kp0 is not None and rhip is not None and lhip is not None and scale > 1e-6:
            midpoint = (np.array(rhip) + np.array(lhip)) / 2.0
            dist = float(np.linalg.norm(np.array(kp0) - midpoint))
            # If the pelvis differs from the midpoint by a non-trivial fraction of body scale, fix it.
            if dist > 0.25 * scale:
                raw[0] = midpoint.tolist()

        # ---------------------------
        # Manual fix: kp 9 and kp 10
        # ---------------------------
        # In the non-original configuration, the head appears "flipped" with respect to vertical axis.
        # We then adjust x and y coordinates to what heuristically appears correct, then adjusts kp 10 accordingly.
        # This is a strong heuristic, but resembles the movements detected in video, which are very different from
        # the ones obtained with standard "non-original" configuration of humanbert.
        if self.enable_keypoint_fixes and ref_neck_like is not None and head_like is not None and scale > 1e-6:
            neck = np.array(ref_neck_like, dtype=np.float32)
            head = np.array(head_like, dtype=np.float32)


            original_head = head.copy()

            # This is your original transformation, kept as-is.
            head[0] = (head[0] - 2 * (neck[0] - head[0])) * 1.1  # Flip x-coordinate with respect to keypoint 8
            head[1] = (head[1] - 2 * (neck[1] - head[1])) / 1.35 # Flip y-coordinate with respect to keypoint 8

            raw[9] = head.tolist()

            # If kp 10 exists, adjust it in a way consistent.
            if forehead_like is not None:
                forehead = np.array(forehead_like, dtype=np.float32)

                translation_x = float(original_head[0] - head[0])
                translation_y = float(original_head[1] - head[1])

                forehead[0] -= translation_x
                forehead[1] -= translation_y

                # Reflect / push relative to the updated head
                forehead[0] = (forehead[0] - 2 * (head[0] - forehead[0]))
                forehead[1] = (forehead[1] - 2 * (head[1] - forehead[1]))

                raw[10] = forehead.tolist()

        # Assemble output in the exact order of keypoint_indices.
        selected_keypoints = [raw[h36m_id] for h36m_id in self.keypoint_indices]
        selected_scores = [raw_scores[h36m_id] for h36m_id in self.keypoint_indices]

        skeleton = {"keypoints": selected_keypoints, "keypoint_scores": selected_scores}
        return skeleton


    def _estimate_body_scale(self, raw: Dict[int, List[float]], scores: Dict[int, float]) -> float:
        """
        Estimate a robust scale for misalignment tests, so heuristics are not hard-coded in raw units.

        Preference:
          - shoulder width (11-14) if both are confident
          - spine length (7-8) if confident
          - fallback: 1.0 (to avoid division by zero)
        """
        def ok(hid: int) -> bool:
            return (hid in raw) and (scores.get(hid, 0.0) >= self.visibility_th)

        if ok(11) and ok(14):
            return float(np.linalg.norm(np.array(raw[11]) - np.array(raw[14])))
        if ok(7) and ok(8):
            return float(np.linalg.norm(np.array(raw[7]) - np.array(raw[8])))
        return 1.0

    def find_main_speaker_skeletons(self) -> List[Dict[str, Any]]:
        """
        Select the main speaker's skeleton over time, considering keypoint distances and scores.
        """
        tracked_skeletons: List[Dict[str, Any]] = []
        selected_skeleton: Dict[str, Any] = {}  # Reference skeleton (selected person's body)

        assert self.poses, "No poses data provided."

        for frame_idx, frame in enumerate(self.poses):
            instances = frame.get("instances", [])
            tracked_person = None

            if not selected_skeleton:
                # Select the main speaker in the first valid frame based on highest mean confidence.
                confidence_list = []
                valid_instances = []

                for inst in instances:
                    sk = self.get_skeleton_from_frame(inst)
                    valid_scores = [s for s in sk["keypoint_scores"] if s > self.visibility_th]
                    if len(valid_scores) >= self.min_valid_keypoints:
                        confidence_list.append(float(np.mean(valid_scores)))
                        valid_instances.append(inst)
                    else:
                        confidence_list.append(-np.inf)

                if confidence_list and len(valid_instances) > 0:
                    max_index = int(np.argmax(confidence_list))
                    if 0 <= max_index < len(valid_instances):
                        selected_skeleton = self.get_skeleton_from_frame(valid_instances[max_index])
                        tracked_person = valid_instances[max_index]

            if selected_skeleton:
                tracked_person = self.get_closest_skeleton(
                    frame_instances=instances,
                    selected_body=selected_skeleton,
                    score_threshold=self.visibility_th,
                    min_valid_keypoints=self.min_valid_keypoints
                )

            if tracked_person:
                skeleton_data = self.get_skeleton_from_frame(tracked_person)

                # actual data has y vertical. Convert it to have z vertical
                skeleton_data = self.convert_to_right_handed_with_z_vertical(skeleton_data)

                # Update the reference skeleton for tracking (keep the original coordinate convention for consistency)
                selected_skeleton = self.get_skeleton_from_frame(tracked_person)
            else:
                # If tracking fails, append an empty dictionary
                skeleton_data = {}

            tracked_skeletons.append(skeleton_data)

        return tracked_skeletons

    def convert_to_right_handed_with_z_vertical(self, skeleton: Dict[str, Any]) -> Dict[str, Any]:
        """TODO check if this is needed also with the skeletons extracted with the original config of MMPose. This is needed with \
         the other configration as the reference system seemed different and some fixes were needed"""
        """
        Converts the skeleton data to a right-handed coordinate system with the z-axis as vertical.

        In the current dataset y is vertical. This conversion swaps y and z:
            (x, y, z) -> (x, z, y)
        """
        keypoints = skeleton["keypoints"]
        for i in range(len(keypoints)):
            x, y, z = keypoints[i]
            keypoints[i] = [x, z, y]
        skeleton["keypoints"] = keypoints
        return skeleton

    def get_closest_skeleton(
        self,
        frame_instances: List[Dict[str, Any]],
        selected_body: Dict[str, Any],
        score_threshold: float = 0.5,
        min_valid_keypoints: int = 5
    ) -> Optional[Dict[str, Any]]:
        """
        Find the closest skeleton to the selected main speaker skeleton based on selected keypoints.
        """
        min_diff = float("inf")
        tracked_person = None

        for inst in frame_instances:
            current = self.get_skeleton_from_frame(inst)
            scores = current["keypoint_scores"]

            valid_keypoints = sum(s >= score_threshold for s in scores)
            if valid_keypoints < min_valid_keypoints:
                continue

            diff = 0.0
            n_valid = 0

            for i, sc in enumerate(scores):
                if sc >= score_threshold and selected_body["keypoint_scores"][i] >= score_threshold:
                    kp_sel = np.array(selected_body["keypoints"][i], dtype=np.float32)
                    kp_cur = np.array(current["keypoints"][i], dtype=np.float32)
                    diff += float(np.linalg.norm(kp_cur - kp_sel))
                    n_valid += 1

            if n_valid > 0:
                avg_diff = diff / n_valid
                if avg_diff < min_diff:
                    min_diff = avg_diff
                    tracked_person = inst

        # Define a base distance threshold based on the selected skeleton
        base_distance = self.calculate_base_distance(selected_body)

        if tracked_person and min_diff <= base_distance:
            return tracked_person
        return None


    def calculate_base_distance(self, skeleton: Dict[str, Any]) -> float:
        """
        Calculate a base distance threshold based on the skeleton's keypoints.
        Here we locate joints by H36M id so it works for both full-body and subsets.
        """
        kp = skeleton.get("keypoints", [])
        if not kp:
            return float("inf")

        m = self._h36m_to_local()

        def get(hid: int) -> Optional[np.ndarray]:
            if hid not in m:
                return None
            return np.array(kp[m[hid]], dtype=np.float32)

        spine_upper = get(8)
        head = get(10)
        left_shoulder = get(11)
        right_shoulder = get(14)

        if spine_upper is None or head is None or left_shoulder is None or right_shoulder is None:
            return float("inf")

        neck_height = float(np.linalg.norm(spine_upper - head))
        shoulder_width = float(np.linalg.norm(left_shoulder - right_shoulder))

        # Base distance as maximum of scaled neck height and shoulder width.
        return max(neck_height * 3.0, shoulder_width * 2.0)


    # ----------------------------
    # Filtering / presence checks
    # ----------------------------

    def keypoints_presence(self) -> bool:
        """
        Calculate the percentage of frames in which each keypoint is present (score >= threshold).

        Note:
        - This runs before interpolation (when we still have keypoint_scores).
        - Empty skeleton dicts are treated as "no keypoints present".
        """
        num_frames = len(self.main_speaker_skeletons)
        if num_frames == 0:
            raise ValueError("There aren't any main speaker skeletons to process!")

        keypoint_presence = {hid: 0 for hid in self.keypoint_indices}

        for skeleton in self.main_speaker_skeletons:
            if not skeleton:
                continue
            kp_scores = skeleton["keypoint_scores"]
            for i, hid in enumerate(self.keypoint_indices):
                if kp_scores[i] >= self.visibility_th:
                    keypoint_presence[hid] += 1

        presence_percent = {hid: (count / num_frames) for hid, count in keypoint_presence.items()}

        for hid, percent in presence_percent.items():
            if percent < self.presence_th:
                return False

        return True


    # ----------------------------
    # Interpolation and smoothing
    # ----------------------------

    def interpolate_keypoints(self) -> Tuple[Optional[List[Dict[str, Any]]], Optional[int], Optional[int]]:
        """
        Fill missing keypoint coordinates by interpolating missing values across frames.
        Interpolation is performed inside the global interval where all keypoints have at least some valid data.

        Returns:
          new_skeletons (list of dict with 'keypoints' only),
          new_start_frame (in original video-frame coords),
          new_end_frame (in original video-frame coords)
        """
        num_frames = len(self.main_speaker_skeletons)
        frame_indices = np.arange(num_frames)

        # Convert minimum duration from seconds to frames (pose-time).
        min_duration_frames = int(self.min_duration_sec * self.pose_fps)

        interpolated = {hid: {"x": np.full(num_frames, np.nan),
                              "y": np.full(num_frames, np.nan),
                              "z": np.full(num_frames, np.nan)}
                        for hid in self.keypoint_indices}

        keypoint_validity = []

        for hid in self.keypoint_indices:
            xs, ys, zs = [], [], []

            for fr in self.main_speaker_skeletons:
                kp_x, kp_y, kp_z = np.nan, np.nan, np.nan
                if fr and "keypoints" in fr and "keypoint_scores" in fr:
                    local_i = self.keypoint_indices.index(hid)
                    sc = fr["keypoint_scores"][local_i]
                    if not np.isnan(sc) and sc >= self.visibility_th:
                        kp_x, kp_y, kp_z = fr["keypoints"][local_i]
                xs.append(kp_x)
                ys.append(kp_y)
                zs.append(kp_z)

            xs = np.array(xs, dtype=np.float32)
            ys = np.array(ys, dtype=np.float32)
            zs = np.array(zs, dtype=np.float32)

            interpolated[hid]["x"] = xs
            interpolated[hid]["y"] = ys
            interpolated[hid]["z"] = zs

            valid_frames = ~np.isnan(xs) & ~np.isnan(ys) & ~np.isnan(zs)
            keypoint_validity.append(valid_frames)

        global_valid_frames = np.all(keypoint_validity, axis=0)
        if not np.any(global_valid_frames):
            raise ValueError("No frames have all keypoints with valid data. Cannot perform interpolation.")

        interp_start = int(np.argmax(global_valid_frames))
        interp_end = int(num_frames - 1 - np.argmax(global_valid_frames[::-1]))

        overall_duration = interp_end - interp_start + 1
        if overall_duration < min_duration_frames:
            print("Overall duration after trimming does not meet minimum requirement. Aborting interpolation.")
            return None, None, None

        # Interpolate each keypoint within [interp_start, interp_end]
        for hid in self.keypoint_indices:
            seg_x = interpolated[hid]["x"][interp_start:interp_end + 1]
            seg_y = interpolated[hid]["y"][interp_start:interp_end + 1]
            seg_z = interpolated[hid]["z"][interp_start:interp_end + 1]

            mask = ~np.isnan(seg_x) & ~np.isnan(seg_y) & ~np.isnan(seg_z)
            if not np.any(mask):
                raise ValueError(f"Keypoint {hid} has no valid data inside interpolation boundaries.")
            if np.sum(mask) < 2:
                raise ValueError(f"Not enough valid points to interpolate keypoint {hid}.")

            # Use pandas linear interpolation (kept from your code; it’s stable and readable).
            seg_x = pd.Series(seg_x).interpolate(method="linear", limit_direction="both").to_numpy()
            seg_y = pd.Series(seg_y).interpolate(method="linear", limit_direction="both").to_numpy()
            seg_z = pd.Series(seg_z).interpolate(method="linear", limit_direction="both").to_numpy()

            if (np.isnan(seg_x) | np.isnan(seg_y) | np.isnan(seg_z)).any():
                raise ValueError(f"Keypoint {hid} still contains NaNs after interpolation.")

            interpolated[hid]["x"][interp_start:interp_end + 1] = seg_x
            interpolated[hid]["y"][interp_start:interp_end + 1] = seg_y
            interpolated[hid]["z"][interp_start:interp_end + 1] = seg_z

        # Assemble new skeletons (keypoints only; scores are no longer needed after interpolation).
        new_skeletons = []
        for j in range(interp_start, interp_end + 1):
            pts = []
            for hid in self.keypoint_indices:
                x = float(interpolated[hid]["x"][j])
                y = float(interpolated[hid]["y"][j])
                z = float(interpolated[hid]["z"][j])
                pts.append([x, y, z])
            new_skeletons.append({"keypoints": pts})

        # Convert pose-frame indices back to original video-frame indices.
        # IMPORTANT CHANGE:
        # Use pose_start_frame_orig (first included pose frame in original coords), not initial_start,
        # to reduce boundary drift when the scene start isn't aligned to the sampling grid.
        new_start_frame = self.pose_start_frame_orig + interp_start * self.downsample_factor
        new_end_frame = self.pose_start_frame_orig + interp_end * self.downsample_factor

        return new_skeletons, int(new_start_frame), int(new_end_frame)

    def smooth_motion(self) -> None:
        """
        Smoothens the motion of keypoints over time using the Savitzky-Golay filter.
        Applies the filter to each coordinate of each keypoint separately.
        """
        window_length = 5  # Must be odd and >= polyorder + 2
        polyorder = 2

        num_keypoints = len(self.keypoint_indices)
        num_coords = 3

        # Build array (frames, joints, 3)
        keypoints_array = np.full((len(self.main_speaker_skeletons), num_keypoints, num_coords), np.nan, dtype=np.float32)

        for frame_idx, skeleton in enumerate(self.main_speaker_skeletons):
            for kp_idx, keypoint in enumerate(skeleton["keypoints"]):
                if keypoint:
                    keypoints_array[frame_idx, kp_idx, :] = np.array(keypoint, dtype=np.float32)

        # Apply filter per joint, per coord.
        for kp_idx in range(num_keypoints):
            for coord in range(num_coords):
                coord_data = keypoints_array[:, kp_idx, coord]
                valid = ~np.isnan(coord_data)

                if not np.any(valid):
                    raise ValueError("No valid data points to smooth for this coordinate.")

                if np.sum(valid) < window_length:
                    # Not enough points, skip smoothing.
                    continue

                smoothed = savgol_filter(coord_data[valid], window_length, polyorder)
                keypoints_array[valid, kp_idx, coord] = smoothed

        # From here on, we only need the numeric array representation.
        self.main_speaker_skeletons = keypoints_array

    # ----------------------------
    # Skeleton structure and kinematics
    # ----------------------------

    def create_parent_indices(self) -> None:
        """
        Creates a list of parent indices showing the parent of each selected keypoint.
        Parent relationships are based on skeleton_info['skeleton_links'] from MMPose dataset_meta.
        """
        if "skeleton_links" not in self.skeleton_info:
            raise ValueError("skeleton_info does not contain 'skeleton_links'. Cannot build parent indices.")

        skeleton_links = self.skeleton_info["skeleton_links"]
        child_to_parent = {child: parent for parent, child in skeleton_links}

        parent_indices = []
        for hid in self.keypoint_indices:
            parent = child_to_parent.get(hid, -1)
            if parent in self.keypoint_indices:
                parent_idx = self.keypoint_indices.index(parent)
            else:
                parent_idx = -1
            parent_indices.append(parent_idx)

        self.parent_indices = parent_indices


    def compute_person_avg_segment_length(self) -> None:
        """
        Computes the average length of each skeletal segment across all frames.
        Saves segment lengths to disk for debugging / reconstruction.
        """
        poses = self.main_speaker_skeletons  # (n_frames, n_joints, 3)
        parents = self.parent_indices

        num_frames, num_joints, _ = poses.shape
        joint_lengths = np.zeros((num_frames, num_joints, 1), dtype=np.float32)

        for f in range(num_frames):
            for j in range(num_joints):
                p = parents[j]
                if p != -1:
                    joint_lengths[f, j, 0] = float(np.linalg.norm(poses[f, j, :] - poses[f, p, :]))

        avg_lengths = np.mean(joint_lengths, axis=0).flatten()
        self.avg_segment_lengths = avg_lengths

        save_path = os.path.join(
            self.save_path,
            "all_motion_data",
            f"person_segments_{self.initial_start}_{self.initial_end}.pkl"
        )
        with open(save_path, "wb") as f:
            pickle.dump(self.avg_segment_lengths, f, protocol=pickle.HIGHEST_PROTOCOL)

    def align_hip_with_global_frame(self, hip_index: int) -> None:
        """
        Aligns the hip with the global frame by translating the motion so the hip is at the origin.
        (Rotation removal is handled separately by remove_orientation_around_z.)
        """
        motion = self.main_speaker_skeletons
        n_frames, n_joints, _ = motion.shape

        hip_pos = motion[:, hip_index]
        new_poses = np.empty_like(motion)

        for f in range(n_frames):
            for j in range(n_joints):
                new_poses[f, j] = motion[f, j] - hip_pos[f]

        self.main_speaker_skeletons = new_poses

    def remove_orientation_around_z(self, shoulder1_ind: int, shoulder2_ind: int, spine1_ind: int, spine2_ind: int, head_ind: int) -> None:
        """
        Removes average global orientation in two steps:
          1) rotate so average spine vector aligns with +Z
          2) rotate around Z so average shoulder vector aligns with +X
        """
        motion = self.main_speaker_skeletons
        n_frames, n_joints, _ = motion.shape

        spine_vec = motion[:, spine2_ind, :] - motion[:, spine1_ind, :]
        avg_spine = np.mean(spine_vec, axis=0)
        norm_spine = np.linalg.norm(avg_spine)
        if norm_spine < 1e-6:
            raise ValueError("Average spine vector is zero.")
        avg_spine /= norm_spine

        target = np.array([0, 0, 1], dtype=np.float32)
        axis = np.cross(avg_spine, target)
        sin_angle = np.linalg.norm(axis)
        cos_angle = float(np.dot(avg_spine, target))
        angle = float(np.arctan2(sin_angle, cos_angle))

        if sin_angle < 1e-6:
            if cos_angle > 0:
                rot_spine = R.identity()
            else:
                arbitrary_axis = np.array([1, 0, 0], dtype=np.float32)
                if np.allclose(avg_spine, arbitrary_axis):
                    arbitrary_axis = np.array([0, 1, 0], dtype=np.float32)
                axis = np.cross(avg_spine, arbitrary_axis)
                rot_spine = R.from_rotvec(np.pi * axis)
        else:
            axis /= sin_angle
            rot_spine = R.from_rotvec(axis * angle)

        motion_flat = motion.reshape(-1, 3)
        rotated = rot_spine.apply(motion_flat).reshape(n_frames, n_joints, 3)

        shoulder_vec = rotated[:, shoulder2_ind, :] - rotated[:, shoulder1_ind, :]
        avg_sh = np.mean(shoulder_vec, axis=0)
        avg_sh[2] = 0.0  # project to XY
        norm_sh = np.linalg.norm(avg_sh)
        if norm_sh < 1e-6:
            raise ValueError("Average shoulder vector is zero after projection.")
        avg_sh /= norm_sh

        angle_sh = float(np.arctan2(avg_sh[1], avg_sh[0]))
        rot_sh = R.from_euler("z", -angle_sh, degrees=False)

        rotated2 = rot_sh.apply(rotated.reshape(-1, 3)).reshape(n_frames, n_joints, 3)
        self.main_speaker_skeletons = rotated2

    def is_skeleton_back(self) -> bool:
        """
        Determines if the skeleton is facing backward based on shoulder ordering.
        If right-shoulder x < left-shoulder x for too many frames, we treat it as back-facing.
        """
        n_bad = 0
        motion = self.main_speaker_skeletons
        scene_length = motion.shape[0]

        ls = self._idx_or_raise(11)
        rs = self._idx_or_raise(14)

        for f in range(scene_length):
            left_sh = motion[f, ls]
            right_sh = motion[f, rs]
            if not np.any(np.isnan(left_sh)) and not np.any(np.isnan(right_sh)):
                if right_sh[0] < left_sh[0]:
                    n_bad += 1

        ratio = n_bad / max(1, scene_length)
        print("BACK", ratio)
        return ratio > 1.0 - self.presence_th

    def is_skeleton_sideways(self) -> bool:
        """
        Determines if the skeleton is sideways based on head/shoulder geometry.

        Note:
        - This is heuristic and dataset-dependent.
        """
        n_bad = 0
        n_valid = 0

        motion = self.main_speaker_skeletons
        scene_length = motion.shape[0]

        spine_upper = self._idx_or_raise(8)
        neck = self._idx_or_raise(9) if 9 in self.keypoint_indices else spine_upper
        head = self._idx_or_raise(10) if 10 in self.keypoint_indices else neck
        ls = self._idx_or_raise(11)
        rs = self._idx_or_raise(14)

        for f in range(scene_length):
            spine_upper_pt = motion[f, spine_upper]
            head_pt = motion[f, head]
            left_sh = motion[f, ls]
            right_sh = motion[f, rs]

            if (np.any(np.isnan(spine_upper_pt)) or np.any(np.isnan(head_pt)) or
                    np.any(np.isnan(left_sh)) or np.any(np.isnan(right_sh))):
                continue

            n_valid += 1

            min_sh_x = min(left_sh[0], right_sh[0])
            max_sh_x = max(left_sh[0], right_sh[0])

            left_len = float(np.linalg.norm(left_sh - spine_upper_pt))
            right_len = float(np.linalg.norm(right_sh - spine_upper_pt))

            if ((head_pt[0] < min_sh_x) or (head_pt[0] > max_sh_x) or
                    (left_len > 1.6 * right_len) or (right_len > 1.6 * left_len)):
                n_bad += 1

        ratio = n_bad / max(1, n_valid)
        print("SIDEWAYS", ratio, "valid_frames", n_valid, "total_frames", scene_length)
        return ratio > 1.0 - self.presence_th


    def compute_rotation_matrices(self) -> np.ndarray:
        """
        Transforms 3D coordinates to rotation matrices between joints.
        Each joint is oriented so that its bone (parent->child) aligns with +Z.
        """
        keypoints_array = self.main_speaker_skeletons
        n_frames, n_joints, _ = keypoints_array.shape
        rot = np.zeros((n_frames, n_joints, 3, 3), dtype=np.float32)

        default_bone = np.array([0, 0, 1], dtype=np.float32)

        for f in range(n_frames):
            for j in range(n_joints):
                p = self.parent_indices[j]
                if p == -1:
                    rot[f, j] = np.eye(3, dtype=np.float32)
                    continue

                direction = keypoints_array[f, j] - keypoints_array[f, p]
                n = float(np.linalg.norm(direction))
                if n > 1e-6:
                    direction = direction / n
                else:
                    direction = default_bone.copy()

                rot[f, j] = self.rotation_between_vectors(default_bone, direction)

        return rot


    def rotation_between_vectors(self, vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
        """
        Returns the rotation matrix rotating vec1 to vec2.
        """
        v = np.cross(vec1, vec2)
        c = float(np.dot(vec1, vec2))
        s = float(np.linalg.norm(v))
        if s == 0:
            return np.eye(3, dtype=np.float32)

        kmat = np.array([[0, -v[2], v[1]],
                         [v[2], 0, -v[0]],
                         [-v[1], v[0], 0]], dtype=np.float32)

        rot = np.eye(3, dtype=np.float32) + kmat + (kmat @ kmat) * ((1.0 - c) / (s ** 2))
        return rot


    def represent_relative_rotations(self, rotation_matrices: np.ndarray) -> np.ndarray:
        """
        Represents rotations as relative rotations between a joint at time t and its parent at time t-1.
        """
        n_frames, n_joints, _, _ = rotation_matrices.shape
        rel = np.zeros_like(rotation_matrices)
        rel[0] = rotation_matrices[0]

        for f in range(1, n_frames):
            for j in range(n_joints):
                p = self.parent_indices[j]
                parent_rot = np.eye(3, dtype=np.float32) if p == -1 else rotation_matrices[f - 1, p]
                rel[f, j] = rotation_matrices[f, j] @ np.linalg.inv(parent_rot)

        return rel


    def np_matrix_to_rotation_6d(self, matrices: np.ndarray) -> np.ndarray:
        """
        Transforms rotation matrices to 6D representation (first two columns).
        Output shape: (n_frames, n_joints, 6)
        """
        return matrices[..., :2, :].copy().reshape(matrices.shape[0], matrices.shape[1], 6)

    def rotation_6d_to_matrix(self, d6: torch.Tensor) -> torch.Tensor:
        """
        Converts 6D rotation representation by Zhou et al. to rotation matrix using Gram-Schmidt orthogonalization.

        Args:
            d6: 6D rotation representation, of shape (..., 6).

        Returns:
            Rotation matrices of shape (..., 3, 3).
        """
        a1, a2 = d6[..., :3], d6[..., 3:]
        b1 = F.normalize(a1, dim=-1)
        b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack((b1, b2, b3), dim=-2)

    # ----------------------------
    # Visualization (optional)
    # ----------------------------

    def set_axes_equal(self, ax) -> None:
        '''
        Set equal aspect ratio for 3D plots.

        :param ax: A matplotlib axis, e.g., as output from plt.gca().
        '''
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        plot_radius = 0.5 * max([x_range, y_range, z_range])
        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


    '''
    def plot_poses_animation(self, path: str, save_name: str, n_frames: int, target_height: float = 1.7) -> None:
        """
        Plot the 3D skeleton animation with uniform human proportions, thicker links,
        and no background/axes. Saves as GIF for debugging.
        """
        # Import matplotlib only if/when needed so the script can run headless.
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        poses = np.array(self.main_speaker_skeletons, dtype=np.float32)  # (frames, joints, 3)

        # Compute current height using robust H36M ids instead of assuming fixed positions.
        m = self._h36m_to_local()
        hip_local = m.get(0, m.get(7, 0))
        head_local = m.get(10, m.get(9, hip_local))

        curr_h = float(np.linalg.norm(poses[0, head_local] - poses[0, hip_local]))
        if curr_h > 1e-6:
            poses = poses * (float(target_height) / curr_h)

        fig = plt.figure(facecolor="none")
        ax = fig.add_subplot(111, projection="3d", facecolor="none")
        ax.set_axis_off()

        lines = []
        for _ in self.parent_indices:
            line, = ax.plot([], [], [], lw=5)
            lines.append(line)

        self.set_axes_equal(ax)

        def update(frame: int):
            for i, parent in enumerate(self.parent_indices):
                if parent == -1:
                    continue
                x = [poses[frame, i, 0], poses[frame, parent, 0]]
                y = [poses[frame, i, 1], poses[frame, parent, 1]]
                z = [poses[frame, i, 2], poses[frame, parent, 2]]
                lines[i].set_data(x, y)
                lines[i].set_3d_properties(z)
            return lines

        anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / max(1e-6, self.pose_fps), blit=False)
        _ensure_dir(path)
        out = os.path.join(path, f"{save_name}.gif")
        anim.save(out, writer="pillow", fps=self.pose_fps)
        plt.close(fig)
    '''

    def plot_poses_animation(self, path: str, save_name: str, n_frames: int, target_height: float = 1.7) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        poses = np.asarray(self.main_speaker_skeletons, dtype=np.float32)  # (frames, joints, 3)
        n_frames = min(int(n_frames), int(poses.shape[0]))

        # Scale to target height (using your existing logic)
        m = self._h36m_to_local()
        hip_local = m.get(0, m.get(7, 0))
        head_local = m.get(10, m.get(9, hip_local))

        curr_h = float(np.linalg.norm(poses[0, head_local] - poses[0, hip_local]))
        if curr_h > 1e-6:
            poses = poses * (float(target_height) / curr_h)

        fig = plt.figure(facecolor="none")
        ax = fig.add_subplot(111, projection="3d", facecolor="none")
        ax.set_axis_off()

        # Make the projection less “weird” (no perspective foreshortening)
        try:
            ax.set_proj_type("ortho")
        except Exception:
            pass

        # Force a consistent front-like camera.
        # With your convention (x left-right, y depth, z up), a front view is achieved
        # by placing the camera on the ±Y axis, looking toward the origin.
        ax.view_init(elev=10, azim=90)  # if mirrored, try azim=-90 or 270

        # --- IMPORTANT: set limits from the data, not from default axes ---
        xyz = poses.reshape(-1, 3)
        xyz = xyz[np.isfinite(xyz).all(axis=1)]  # safety

        # Robust bounds: ignore extreme outliers
        lo = np.quantile(xyz, 0.02, axis=0)
        hi = np.quantile(xyz, 0.98, axis=0)

        center = 0.5 * (lo + hi)

        # Use the biggest robust span as the "diameter"
        span = hi - lo
        radius = 0.45 * float(np.max(span))  # 0.55 adds a bit of margin; reduce for tighter zoom

        # Optional: don’t let it zoom out beyond a human-ish box
        radius = min(radius, 2.0 * target_height)  # clamp huge outliers
        radius = max(radius, 0.8 * target_height)  # avoid over-zooming in

        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)

        lines = []
        for _ in self.parent_indices:
            line, = ax.plot([], [], [], lw=5)
            lines.append(line)

        def update(frame: int):
            for i, parent in enumerate(self.parent_indices):
                if parent == -1:
                    continue
                x = [poses[frame, i, 0], poses[frame, parent, 0]]
                y = [poses[frame, i, 1], poses[frame, parent, 1]]
                z = [poses[frame, i, 2], poses[frame, parent, 2]]
                lines[i].set_data(x, y)
                lines[i].set_3d_properties(z)
            return lines

        anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / max(1e-6, self.pose_fps), blit=False)
        _ensure_dir(path)
        out = os.path.join(path, f"{save_name}.gif")
        anim.save(out, writer="pillow", fps=self.pose_fps)
        plt.close(fig)



# ------------------------------------------------------------
# Small utilities
# ------------------------------------------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _parse_int_list(csv: str) -> List[int]:
    csv = csv.strip()
    if not csv:
        return []
    return [int(x.strip()) for x in csv.split(",") if x.strip()]


def _safe_open_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_pose_pickle(pkl_path: str) -> Dict[str, Any]:
    """
    Loads pose pickle and performs minimal validation.
    Works with both _mmpose_data_output.pkl and _mmpose_data_output_final.pkl as long as
    each instance contains 'keypoints' and 'keypoint_scores'.
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict) or "meta_info" not in data or "instance_info" not in data:
        raise ValueError(f"Pose pickle at {pkl_path} does not have expected keys (meta_info, instance_info).")

    if not isinstance(data["instance_info"], list):
        raise ValueError(f"Pose pickle at {pkl_path}: instance_info must be a list.")

    # Optional quick consistency check for the first non-empty frame.
    for fr in data["instance_info"][:50]:
        inst = fr.get("instances", [])
        if inst:
            if "keypoints" not in inst[0] or "keypoint_scores" not in inst[0]:
                raise ValueError(
                    f"Pose pickle at {pkl_path}: instances must contain keypoints and keypoint_scores."
                )
            break

    return data


def build_frame_index(instance_info: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """
    Index frames by frame_id for fast slicing.
    Assumes frame_id is integer. Supports either 0-based or 1-based frame_id.
    """
    idx = {}
    for fr in instance_info:
        if "frame_id" not in fr:
            continue
        idx[int(fr["frame_id"])] = fr
    return idx


def infer_pose_sampling(video_fps: float, meta_info: Dict[str, Any], fallback_pose_fps: float) -> Tuple[float, int]:
    """
    Returns:
      pose_fps_effective (float)  - the actual pose FPS used by extraction
      downsample_factor (int)     - approx how many video frames per pose frame we used during poses extraction.
                                    This is needed as poses were not extracted so that they are perfectly aligned with
                                    other modalities, but by doing max(1, int(round(video_fps / pose_fps))). For perfect
                                    alignment with other modalities, we need to upsample through interpolation

    In the files _video_mmpose_data_output_final, meta_info should include pose_fps, that is the fps we want to have for our motion.
    Original mmpose pickles _video_mmpose_data_output don't contain pose_fps, so we use a default value.
    """
    pose_fps = float(meta_info.get("pose_fps", fallback_pose_fps))
    if pose_fps <= 0:
        pose_fps = float(fallback_pose_fps)

    # In the extraction pipeline, frames are skipped by:
    #   skip = round(video_fps / target_processing_fps)
    # pose_fps_effective is video_fps / skip.
    # If we have pose_fps_effective, we reconstruct skip by round(video_fps / pose_fps).
    downsample_factor = max(1, int(round(video_fps / pose_fps)))

    # Recompute pose_fps_effective from this integer skip to remain consistent with frame_id mapping.
    pose_fps_effective = float(video_fps / downsample_factor)
    return pose_fps_effective, downsample_factor


def scene_to_pose_id_range(scene_start: int, scene_end: int, downsample_factor: int) -> Tuple[int, int, int]:
    """
    Converts a scene [start_frame, end_frame) in *video-frame coordinates* to an interval of pose frame_ids.
    This is needed to pick the right poses referred to a scene in the video. Not all the scenes contain good motion.

    Important subtlety:
      pose frame_id in your MMPose dump is 1-based, and corresponds to original frame:
        orig_frame = (frame_id - 1) * downsample_factor

    Condition for being inside the scene:
        scene_start <= (frame_id - 1)*downsample_factor < scene_end

    This yields:
        frame_id >= ceil(scene_start / downsample_factor) + 1
        frame_id <  ceil(scene_end   / downsample_factor) + 1

    Returns:
      start_id (inclusive), end_id_exclusive, pose_start_orig_frame
    """
    if downsample_factor <= 0:
        downsample_factor = 1

    start_id = int(math.ceil(scene_start / downsample_factor) + 1)
    end_id_excl = int(math.ceil(scene_end / downsample_factor) + 1)

    pose_start_orig = (start_id - 1) * downsample_factor
    return start_id, end_id_excl, pose_start_orig


# ------------------------------------------------------------
# Scene selection and per-video processing
# ------------------------------------------------------------

#It divides the scene using sliding windows and stride
def split_scene_into_windows(scene, fps, window_duration_sec=5, stride_sec=.5):
    window_duration_frames = int(window_duration_sec * fps)
    stride_frames = int(stride_sec * fps)
    windows = []
    start_frame = scene['start_frame']
    end_frame = scene['end_frame']
    current_start = start_frame
    while current_start + window_duration_frames <= end_frame:
        window = {
            'start_frame': current_start,
            'end_frame': current_start + window_duration_frames
        }
        windows.append(window)
        current_start += stride_frames
    return windows

# Takes only scenes that last longer than a certain duration
def select_long_scenes(scenes: List[Dict[str, Any]], fps: float, min_duration_seconds: float = 5.0) -> List[Dict[str, Any]]:
    min_frames = int(min_duration_seconds * fps)
    out = []
    for s in scenes:
        dur = int(s["end_frame"]) - int(s["start_frame"])
        if dur >= min_frames:
            out.append(s)
    return out

def process_video_data(
    scenes: List[Dict[str, Any]],
    motion_data: Dict[str, Any],
    keypoint_indices: List[int],
    video_folder: str,
    args
) -> None:
    """
    Processes all eligible scenes of one video folder using the given pose pickle.
    """
    print("Motion data keys:", motion_data.keys())
    meta = motion_data["meta_info"]

    # The "real_fps" is the video FPS stored in your final pickle meta_info['fps'].
    # If it's missing (older pickles), we require the CLI --video-fps override.
    if "fps" in meta:
        real_fps = float(meta["fps"])
    else:
        if args.video_fps is None:
            raise ValueError("meta_info['fps'] missing in pose pickle. Provide --video-fps.")
        real_fps = float(args.video_fps)

    pose_fps_effective, downsample_factor = infer_pose_sampling(real_fps, meta, args.pose_fps)
    print(f"[Meta] fps={real_fps:.3f}, pose_fps_effective≈{pose_fps_effective:.3f}, downsample_factor={downsample_factor}")

    # Use pose_fps_effective as the internal pose_fps. This matters for time-based thresholds.
    pose_fps_for_processing = pose_fps_effective

    long_scenes = select_long_scenes(scenes, real_fps, min_duration_seconds=args.min_scene_duration_sec)
    print(f"[Scenes] Loaded={len(scenes)}, selected_long={len(long_scenes)}")

    frame_map = build_frame_index(motion_data["instance_info"])

    # Detect whether frame_id is 0-based or 1-based (that is, if fram indexing starts by counting from 0 or 1)
    # Current MMPose pipeline uses 1-based; some others might be 0-based.
    frame_ids = sorted(frame_map.keys())
    if not frame_ids:
        raise ValueError("Pose pickle has no frame_id entries.")
    frame_id_min = frame_ids[0]
    frame_id_shift = 1 if frame_id_min == 0 else 0

    for scene in long_scenes:
        s0 = int(scene["start_frame"])
        s1 = int(scene["end_frame"])

        start_id, end_id_excl, pose_start_orig = scene_to_pose_id_range(s0, s1, downsample_factor)
        start_id += frame_id_shift
        end_id_excl += frame_id_shift
        pose_start_orig = int((start_id - frame_id_shift - 1) * downsample_factor)

        # Pull frames in increasing pose frame_id order.
        scene_frames = []
        for fid in range(start_id, end_id_excl):
            fr = frame_map.get(fid, None)
            if fr is not None:
                scene_frames.append(fr)

        if not scene_frames:
            print(f"[Scene] No pose frames found for scene {s0}-{s1} (pose ids {start_id}-{end_id_excl}).")
            continue

        print(f"[Scene] {s0}-{s1}: pose_frames={len(scene_frames)}  (pose_id_range [{start_id}, {end_id_excl}))")

        processor = MotionPreprocessor(
            poses=scene_frames,
            scene=(s0, s1),
            scene_fps=real_fps,
            keypoint_indices=keypoint_indices,
            visibility_th=args.visibility_th,
            presence_th=args.presence_th,
            min_valid_keypoints=args.min_valid_keypoints,
            pose_fps=pose_fps_for_processing,
            min_duration_sec=args.min_duration_sec,
            skeleton_info=meta,
            save_path=video_folder,
            pose_start_frame_orig=pose_start_orig,
            enable_debug_gifs=(not args.no_debug_gifs),
            enable_keypoint_fixes=(not args.disable_keypoint_fixes),
        )

        result, message = processor.process_scene_motion()
        if not result:
            print(f"[Scene] Rejected: {message}")


def find_default_files_in_video_folder(video_folder: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Looks for:
      - a pose pickle like *_mmpose_data_output_final.pkl (preferred) or *_mmpose_data_output.pkl
      - a scene JSON like *_scenes.json

    Returns (pose_pkl_path, scenes_json_path).
    """
    pose_pkl = None
    scenes_json = None

    # Prefer the _final file (it includes bbox info and fixed meta keys in your pipeline).
    for fn in os.listdir(video_folder):
        if fn.endswith("_mmpose_data_output_final.pkl"):
            pose_pkl = os.path.join(video_folder, fn)
            break

    if pose_pkl is None:
        for fn in os.listdir(video_folder):
            if fn.endswith("_mmpose_data_output.pkl"):
                pose_pkl = os.path.join(video_folder, fn)
                break

    for fn in os.listdir(video_folder):
        if fn.endswith("_scenes.json"):
            scenes_json = os.path.join(video_folder, fn)
            break

    return pose_pkl, scenes_json


def iter_video_folders(playlists_folder: str) -> List[str]:
    """
    Traverse the TED folder structure:
      playlists_folder/<playlist>/<video_folder>/
    and return all video_folder paths.
    """
    out = []
    playlists_folder = os.path.abspath(playlists_folder)

    for playlist in os.listdir(playlists_folder):
        p_path = os.path.join(playlists_folder, playlist)
        if not os.path.isdir(p_path):
            continue
        for video_folder in os.listdir(p_path):
            v_path = os.path.join(p_path, video_folder)
            if os.path.isdir(v_path):
                out.append(v_path)
    return out

def build_arg_parser() -> ArgumentParser:
    p = ArgumentParser(description="Process previously generated pose pickles into motion (6D rotations).")

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--video-folder", type=str, default="", help="Process a single video folder containing poses + scenes.")
    mode.add_argument("--playlists-folder", type=str, default="", help="Process all video folders inside playlists folder.")
    mode.add_argument("--pose-pkl", type=str, default="", help="Direct path to a pose pickle file.")
    # When --pose-pkl is used, you must also pass --scenes-json and --output-folder.
    p.add_argument("--scenes-json", type=str, default="", help="Direct path to scenes JSON (required with --pose-pkl).")
    p.add_argument("--output-folder", type=str, default="", help="Where to save outputs (required with --pose-pkl).")

    # Pose sampling defaults (used only when meta_info lacks fields).
    p.add_argument("--pose-fps", type=float, default=15.0, help="Fallback pose FPS if meta_info['pose_fps'] missing.")
    p.add_argument("--video-fps", type=float, default=None, help="Fallback video FPS if meta_info['fps'] missing.")

    # Scene / motion filtering thresholds.
    p.add_argument("--min-scene-duration-sec", type=float, default=5.0, help="Minimum scene duration to process.")
    p.add_argument("--min-duration-sec", type=float, default=5.0, help="Minimum duration after trimming/interpolation.")

    p.add_argument("--visibility-th", type=float, default=0.4, help="Keypoint confidence threshold for validity.")
    p.add_argument("--presence-th", type=float, default=0.90, help="Fraction of frames each keypoint must be present.")
    p.add_argument("--min-valid-keypoints", type=int, default=5, help="Min keypoints above visibility_th per skeleton.")

    # Keypoints selection.
    p.add_argument(
        "--keypoint-indices",
        type=str,
        default="7,8,9,14,15,16,11,12,13",
        help="Comma-separated H36M keypoint ids to use. Default: the 9 upper-body joints used in the paper.",
    )

    # Debug controls.
    p.add_argument("--no-debug-gifs", action="store_true", help="Disable saving GIF animations for debugging.")
    p.add_argument("--disable-keypoint-fixes", action="store_true", default=False,
                   help="Disable the heuristic fixes for keypoints 0/9/10 (even when misalignment detected).")

    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    keypoint_indices = _parse_int_list(args.keypoint_indices)
    if not keypoint_indices:
        raise ValueError("Empty --keypoint-indices. Provide at least one keypoint id.")

    if args.pose_pkl:
        if not args.scenes_json or not args.output_folder:
            raise ValueError("With --pose-pkl you must also specify --scenes-json and --output-folder.")
        pose_path = os.path.abspath(args.pose_pkl)
        scenes_path = os.path.abspath(args.scenes_json)
        out_folder = os.path.abspath(args.output_folder)

        motion_data = load_pose_pickle(pose_path)
        scenes_data = _safe_open_json(scenes_path)
        if not isinstance(scenes_data, list):
            raise ValueError("Scenes JSON must contain a list of scenes.")

        print(f"[Input] pose_pkl={pose_path}")
        print(f"[Input] scenes_json={scenes_path}")
        print(f"[Output] folder={out_folder}")

        process_video_data(scenes_data, motion_data, keypoint_indices, out_folder, args)
        return

    if args.video_folder:
        video_folder = os.path.abspath(args.video_folder)
        pose_pkl, scenes_json = find_default_files_in_video_folder(video_folder)

        if pose_pkl is None:
            raise FileNotFoundError(f"No pose pickle found in {video_folder} (expected *_mmpose_data_output*.pkl).")
        if scenes_json is None:
            raise FileNotFoundError(f"No scenes JSON found in {video_folder} (expected *_scenes.json).")

        motion_data = load_pose_pickle(pose_pkl)
        scenes_data = _safe_open_json(scenes_json)

        print(f"[Folder] {video_folder}")
        print(f"[Using] pose_pkl={pose_pkl}")
        print(f"[Using] scenes_json={scenes_json}")

        process_video_data(scenes_data, motion_data, keypoint_indices, video_folder, args)
        return

    if args.playlists_folder:
        playlists_folder = os.path.abspath(args.playlists_folder)
        folders = iter_video_folders(playlists_folder)
        print(f"[Playlists] Found {len(folders)} video folders under {playlists_folder}")

        for i, video_folder in enumerate(folders, 1):
            pose_pkl, scenes_json = find_default_files_in_video_folder(video_folder)
            if pose_pkl is None or scenes_json is None:
                print(f"[Skip] Missing pose pickle or scenes json in {video_folder}")
                continue

            try:
                motion_data = load_pose_pickle(pose_pkl)
                scenes_data = _safe_open_json(scenes_json)

                print(f"\n[{i}/{len(folders)}] Processing folder: {video_folder}")
                process_video_data(scenes_data, motion_data, keypoint_indices, video_folder, args)

            except Exception as e:
                # Keep running on other videos. This is useful in large datasets.
                print(f"[Error] {video_folder}: {repr(e)}")

        return

if __name__ == "__main__":
    main()
