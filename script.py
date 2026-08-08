

import cv2
from pathlib import Path

VIDEO_PATH = Path("data/Drone.mp4")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

FRAME_NUMBER = 40
RESIZE_WIDTH = 640


def save_single_frame():
    cap = cv2.VideoCapture(str(VIDEO_PATH))

    cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_NUMBER)
    success, frame = cap.read()

    if not success:
        print("Could not read frame", FRAME_NUMBER)
        cap.release()
        return

    h, w = frame.shape[:2]
    scale = RESIZE_WIDTH / w
    new_h = int(h * scale)
    frame = cv2.resize(frame, (RESIZE_WIDTH, new_h), interpolation=cv2.INTER_AREA)

    output_path = OUTPUT_DIR / f"drone_frame_{FRAME_NUMBER}.png"
    cv2.imwrite(str(output_path), frame)

    cap.release()
    print("Saved", output_path)


if __name__ == "__main__":
    save_single_frame()