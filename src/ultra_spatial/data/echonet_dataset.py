import random
from torch.utils.data import Dataset
from .video_io import list_videos, read_video_frames
from .transforms import resize_center_crop, to_tensor, apply_synthetic_degradation


class EchoNetClips(Dataset):
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
        self.root = root
        self.split = split
        self.frames_per_clip = frames_per_clip
        self.frame_size = frame_size
        self.fps_subsample = fps_subsample
        self.grayscale = grayscale
        self.synth_cfg = synth_cfg
        random.seed(seed)
        videos = list_videos(root)
        if len(videos) == 0:
            raise FileNotFoundError(f"No videos under {root}/videos")
        vids_sorted = sorted(videos)
        n = len(vids_sorted)
        n_train = int(0.7 * n)
        n_val = int(0.15 * n)
        if split == "train":
            self.videos = vids_sorted[:n_train]
        elif split == "val":
            self.videos = vids_sorted[n_train : n_train + n_val]
        else:
            self.videos = vids_sorted[n_train + n_val :]

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        path = self.videos[idx]
        frames = read_video_frames(
            path, fps_subsample=self.fps_subsample, grayscale=self.grayscale
        )
        if len(frames) < self.frames_per_clip:
            k = self.frames_per_clip - len(frames)
            frames = frames + frames[:k]
        start = random.randint(0, max(0, len(frames) - self.frames_per_clip))
        clip = frames[start : start + self.frames_per_clip]
        clip = [resize_center_crop(f, self.frame_size) for f in clip]
        clean = to_tensor(clip)
        degraded, meta = apply_synthetic_degradation(clean, self.synth_cfg)
        return {"degraded": degraded, "clean": clean, "meta": meta, "path": path}
