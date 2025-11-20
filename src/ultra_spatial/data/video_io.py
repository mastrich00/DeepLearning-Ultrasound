import cv2, os
from glob import glob


def list_videos(root):
    return sorted(
        glob(os.path.join(root, "videos", "*.mp4"))
        + glob(os.path.join(root, "videos", "*.avi"))
    )


def read_video_frames(path, max_frames=None, fps_subsample=1, grayscale=True):
    cap = cv2.VideoCapture(path)
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % fps_subsample == 0:
            frame = (
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if grayscale
                else cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )
            frames.append(frame)
        idx += 1
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return frames
