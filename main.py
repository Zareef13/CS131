import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


DATA_PATH = Path("data/car.MP4")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

MAX_FRAMES = None
RESIZE_WIDTH = 640


def normalize_map(x):
    x = x.astype(np.float32)

    min_val = np.min(x)
    max_val = np.max(x)

    normalized = (x - min_val) / (max_val - min_val + 1e-8)

    return normalized


def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray, blurred


def load_video_frames(video_path):
    cap = cv2.VideoCapture(str(video_path))
    frames = []

    while True:
        if MAX_FRAMES is not None and len(frames) >= MAX_FRAMES:
            break

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        h, w = frame.shape[:2]
        scale = RESIZE_WIDTH / w
        new_h = int(h * scale)
        frame = cv2.resize(frame, (RESIZE_WIDTH, new_h), interpolation=cv2.INTER_AREA)

        frames.append(frame)

    cap.release()
    return frames


def compute_structure_scores(frame):
    gray, blurred = preprocess_frame(frame)

    canny_edges = cv2.Canny(blurred, 50, 150)

    edge_map = normalize_map(canny_edges)
    edge_map = cv2.dilate(edge_map, np.ones((5, 5), np.uint8), iterations=1)
    edge_map = cv2.GaussianBlur(edge_map, (11, 11), 0)
    edge_map = normalize_map(edge_map)

    detected_lines = cv2.HoughLinesP(
        canny_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=60,
        maxLineGap=10,
    )

    line_map = np.zeros_like(edge_map)

    if detected_lines is not None:

        for line in detected_lines:
            points = line[0]

            x1 = points[0]
            y1 = points[1] 
            x2 = points[2]
            y2 = points[3]

            cv2.line(line_map, (x1, y1), (x2, y2), 1.0, 2)

    line_map = cv2.GaussianBlur(line_map, (9, 9), 0)
    line_map = normalize_map(line_map)

    return gray, canny_edges, edge_map, line_map


def compute_motion_score(gray1, gray2):
    flow = cv2.calcOpticalFlowFarneback(
        gray1,
        gray2,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=21,
        iterations=3,
        poly_n=7,
        poly_sigma=1.5,
        flags=0,
    )

    flow_x = flow[:, :, 0]
    flow_y = flow[:, :, 1]

    magnitude = np.sqrt(flow_x ** 2 + flow_y ** 2)

    motion_score = normalize_map(magnitude)
    motion_score = cv2.GaussianBlur(motion_score, (11, 11), 0)
    motion_score = normalize_map(motion_score)

    return motion_score


def compute_hybrid_risk(frame1, frame2):
    gray1, canny_edges, edge_map, line_map = compute_structure_scores(frame1)
    gray2, blurred2 = preprocess_frame(frame2)

    motion_score = compute_motion_score(gray1, gray2)

    motion_part = 0.70 * motion_score
    edge_part = 0.20 * edge_map
    line_part = 0.10 * line_map

    risk = motion_part + edge_part + line_part

    threshold = np.percentile(risk, 45)

    risk[risk < threshold] = 0

    risk_display = cv2.GaussianBlur(risk, (41, 41), 0)
    risk_display = normalize_map(risk_display)

    heatmap = cv2.applyColorMap((risk_display * 255).astype(np.uint8), cv2.COLORMAP_HOT)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(frame1, 0.50, heatmap, 0.40, 0)

    return {
        "overlay": overlay,
        "risk_display": risk_display,
        "motion_score": motion_score,
        "edge_map": edge_map,
        "line_map": line_map,
        "canny_edges": canny_edges,
    }


def save_sample_frames(frames):
    indices = [ 0, 40, 80, 120 , 160]
    num_images = len(indices)

    fig, axes = plt.subplots(1, num_images, figsize=(4 * num_images, 5))

    for i in range(num_images):
        idx = indices[i]

        if idx >= len(frames):
            idx = len(frames) - 1

        axes[i].imshow(frames[idx])
        axes[i].set_title(f"Frame {idx}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sample_frames.png")
    plt.close()


def save_edge_demo(frames):
    frame = frames[min(40, len(frames) - 1)]
    gray, blurred = preprocess_frame(frame)
    edges = cv2.Canny(blurred, 50, 150)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(frame)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(blurred, cmap="gray")
    axes[1].set_title("Grayscale + Blur")
    axes[1].axis("off")

    axes[2].imshow(edges, cmap="gray")
    axes[2].set_title("Canny Edges")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "edge_demo.png")
    plt.close()


def save_hough_demo(frames):
    frame = frames[min(40, len(frames) - 1)]
    gray, blurred = preprocess_frame(frame)
    edges = cv2.Canny(blurred, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=60,
        maxLineGap=10,
    )

    line_image = frame.copy()

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(frame)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(edges, cmap="gray")
    axes[1].set_title("Canny Edges")
    axes[1].axis("off")

    axes[2].imshow(line_image)
    axes[2].set_title("Hough Lines")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "hough_demo.png")
    plt.close()


def save_optical_flow_demo(frames):
    idx = min(40, len(frames) - 2)
    frame1 = frames[idx]
    frame2 = frames[idx + 1]

    gray1, _ = preprocess_frame(frame1)
    gray2, _ = preprocess_frame(frame2)
    motion_score = compute_motion_score(gray1, gray2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(frame1)
    axes[0].set_title(f"Frame {idx}")
    axes[0].axis("off")

    axes[1].imshow(frame2)
    axes[1].set_title(f"Frame {idx + 1}")
    axes[1].axis("off")

    axes[2].imshow(motion_score, cmap="hot")
    axes[2].set_title("Optical Flow Magnitude")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "optical_flow_demo.png")
    plt.close()


def save_risk_overlay_demo(frames):
    idx = min(40, len(frames) - 2)
    result = compute_hybrid_risk(frames[idx], frames[idx + 1])

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(frames[idx])
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(result["motion_score"], cmap="hot")
    axes[1].set_title("Motion Score")
    axes[1].axis("off")

    axes[2].imshow(result["risk_display"], cmap="hot")
    axes[2].set_title("Hybrid Risk Map")
    axes[2].axis("off")

    axes[3].imshow(result["overlay"])
    axes[3].set_title("Hybrid Risk Overlay")
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "risk_overlay_frame_40.png")
    plt.close()


def save_risk_overlay_video(frames):
    if len(frames) < 2:
        print("Need at least 2 frames to create risk overlay video.")
        return

    h, w = frames[0].shape[:2]
    output_path = OUTPUT_DIR / "risk_overlay_video.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, 20, (w, h))

    if not writer.isOpened():
        print("VideoWriter failed to open.")
        return

    for i in range(len(frames) - 1):
        result = compute_hybrid_risk(frames[i], frames[i + 1])

        overlay_bgr = cv2.cvtColor(result["overlay"], cv2.COLOR_RGB2BGR)
        writer.write(overlay_bgr)

        if i % 50 == 0:
            print(f"Processed frame {i}/{len(frames) - 1}")

    writer.release()


def main():
    frames = load_video_frames(DATA_PATH)

    print("Loaded frames:", len(frames))
    print("Starting processing...")

    if len(frames) == 0:
        print("No frames loaded. Check the video path.")
        return

    print("Frame shape:", frames[0].shape)

    save_sample_frames(frames)
    print("Saved outputs/sample_frames.png")

    save_edge_demo(frames)
    print("Saved outputs/edge_demo.png")

    save_hough_demo(frames)
    print("Saved outputs/hough_demo.png")

    save_optical_flow_demo(frames)
    print("Saved outputs/optical_flow_demo.png")

    save_risk_overlay_demo(frames)
    print("Saved outputs/risk_overlay_frame_40.png")

    save_risk_overlay_video(frames)
    print("Saved outputs/risk_overlay_video.mp4")


if __name__ == "__main__":
    main()