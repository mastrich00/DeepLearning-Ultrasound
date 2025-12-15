# src/ultra_spatial/data/echonet_dataset.py
"""
Dataset wrapper for EchoNet-style video files producing fixed-length clips.

This Dataset:
 - Lists video files under <root>/videos (via list_videos).
 - Reads frames with read_video_frames and applies center-crop + resize.
 - Forms fixed-length clips by random temporal cropping (with simple wrap padding).
 - Converts frames to torch tensors with shape [T, C, H, W].
 - Applies the synthetic-degradation pipeline to produce (degraded, clean) pairs.
 - Returns a dict with keys: "degraded", "clean", "meta", "path".

Notes:
 - Splitting is simple video-wise 70/15/15 by sorted filename order. Change
   this if patient-wise splits are required.
 - The synthetic config `synth_cfg` should match the expected schema used by
   apply_synthetic_degradation in transforms.py.
"""

import random
from torch.utils.data import Dataset
from .video_io import list_videos, read_video_frames
from .transforms import resize_center_crop, to_tensor, apply_synthetic_degradation


class EchoNetClips(Dataset):
    """
    EchoNetClips dataset.

    Args:
      root (str): root folder containing a "videos" subfolder with .avi/.mp4 files.
      split (str): one of "train", "val", "test" controlling which subset to use.
      frames_per_clip (int): number of frames in each returned clip (T).
      frame_size (int): spatial size to resize frames to (square).
      fps_subsample (int): keep every N-th frame when reading the video.
      grayscale (bool): convert frames to grayscale when reading.
      synth_cfg (dict): configuration for synthetic degradation pipeline.
      seed (int): seed for deterministic video ordering and temporal crop.
    """

    def __init__(
        self,
        root,
        split="train",
        frames_per_clip=16,
        frame_size=128,
        fps_subsample=2,
        grayscale=True,
        synth_cfg=None,
        seed=1337,
    ):
        # store basic params
        self.root = root
        self.split = split
        self.frames_per_clip = frames_per_clip
        self.frame_size = frame_size
        self.fps_subsample = fps_subsample
        self.grayscale = grayscale
        self.synth_cfg = synth_cfg

        # seed ensures reproducible slicing / ordering across runs
        random.seed(seed)

        # list available videos; expected folder layout: <root>/videos/*.avi|*.mp4
        videos = list_videos(root)
        if len(videos) == 0:
            raise FileNotFoundError(f"No videos under {root}/videos")

        # simple deterministic split: sort filenames and split 70/15/15
        vids_sorted = sorted(videos)
        n = len(vids_sorted)
        n_train = int(0.7 * n)
        n_val = int(0.15 * n)
        if split == "train":
            self.videos = vids_sorted[:n_train]
        elif split == "val":
            self.videos = vids_sorted[n_train : n_train + n_val]
        else:  # "test" or other
            self.videos = vids_sorted[n_train + n_val :]

    def __len__(self):
        # number of video files in the selected split
        return len(self.videos)

    def __getitem__(self, idx):
        """
        Return one sample dict:
          {
            "degraded": Tensor [T, C, H, W],
            "clean":     Tensor [T, C, H, W],
            "meta":      dict with synthesis params,
            "path":      source video path
          }

        Steps:
          1. Read all frames from the file with fps_subsample and grayscale options.
          2. If the video has fewer than frames_per_clip, repeat frames from the start
             to reach the requested length (simple wrap-padding).
          3. Randomly pick a contiguous temporal window of length frames_per_clip.
          4. Center-crop + resize every frame to (frame_size, frame_size).
          5. Stack frames into a tensor [T, C, H, W] and normalize to [0,1].
          6. Apply synthetic degradation to get (degraded, meta).
        """
        path = self.videos[idx]

        # read decoded frames as NumPy arrays; read_video_frames applies fps_subsample
        frames = read_video_frames(
            path, fps_subsample=self.fps_subsample, grayscale=self.grayscale
        )

        # If there are fewer frames than requested, wrap-pad by repeating the start frames.
        # This simple strategy avoids failing on very short videos.
        if len(frames) < self.frames_per_clip:
            k = self.frames_per_clip - len(frames)
            frames = frames + frames[:k]

        # choose a random start index for the clip (uniform over feasible starts)
        start = random.randint(0, max(0, len(frames) - self.frames_per_clip))
        clip = frames[start : start + self.frames_per_clip]

        # spatial preprocessing: center-crop to square and resize to frame_size
        clip = [resize_center_crop(f, self.frame_size) for f in clip]

        # convert list of frames to torch tensor [T, C, H, W], floats in [0,1]
        clean = to_tensor(clip)

        # apply synthetic degradation pipeline to create paired supervision
        degraded, meta = apply_synthetic_degradation(clean, self.synth_cfg)

        # return dictionary used throughout training code
        return {"degraded": degraded, "clean": clean, "meta": meta, "path": path}
