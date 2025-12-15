# src/ultra_spatial/data/video_io.py
"""
Small utilities for enumerating and reading video files used by the dataset loader.

Notes:
- Uses OpenCV (cv2) for decoding. OpenCV returns frames in BGR channel order for color.
- The functions return lists of NumPy arrays (dtype uint8) representing images.
- For large collections or long videos consider streaming/iterator-based reading to avoid high memory use.
"""

import cv2
import os
from glob import glob


def list_videos(root):
    """
    List video files under <root>/videos with common extensions.

    Args:
        root (str): path to dataset root directory. The function looks for a
                    subfolder named "videos" and returns files inside it.

    Returns:
        list[str]: sorted list of full paths to video files with .mp4 or .avi extensions.

    Example:
        >>> list_videos("/data/echonet")
        ['/data/echonet/videos/0001.avi', '/data/echonet/videos/0002.mp4', ...]
    """
    return sorted(
        glob(os.path.join(root, "videos", "*.mp4"))
        + glob(os.path.join(root, "videos", "*.avi"))
    )


def read_video_frames(path, max_frames=None, fps_subsample=1, grayscale=True):
    """
    Read frames from a video file using OpenCV.

    Parameters:
      path (str): path to a video file readable by OpenCV.
      max_frames (int|None): optional cap on number of frames to return. If None,
                             read until the end of the file or until other limits apply.
      fps_subsample (int): keep only every `fps_subsample`-th decoded frame. Use 1
                           to keep all frames, 2 to keep every second frame, etc.
      grayscale (bool): if True convert frames to single-channel grayscale via
                        BGR->GRAY conversion. If False, convert to RGB color ordering.

    Returns:
      list[np.ndarray]: list of frames. Each frame is a NumPy array with shape:
        - grayscale: [H, W] (uint8)
        - color : [H, W, 3] in RGB ordering if grayscale=False

    Implementation details and caveats:
      - OpenCV's VideoCapture returns frames in BGR order; when grayscale=False we
        convert to RGB because most PyTorch/image code expects RGB ordering.
      - The function reads sequentially and applies the fps_subsample filter by
        counting decoded frames. This is simple and robust across codecs.
      - The caller should handle resizing, normalization, and augmentation later.
      - For very large videos or many files this function may consume a lot of memory;
        use streaming or a DataLoader that reads frames on demand in production.
    """
    cap = cv2.VideoCapture(path)
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            # end of file or read error
            break

        # apply temporal subsampling: keep only frames where idx % fps_subsample == 0
        if idx % fps_subsample == 0:
            # convert BGR->GRAY or BGR->RGB depending on requested output
            frame = (
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if grayscale
                else cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )
            frames.append(frame)

        idx += 1

        # stop early if we've collected enough frames
        if max_frames and len(frames) >= max_frames:
            break

    cap.release()
    return frames
