import os
import numpy as np
import cv2

class Config:
    current_path = os.path.dirname(os.path.abspath(__file__))
    rgb_path = os.path.join(current_path, "data", "rgb_data.bin")

# 이미지 크기 및 채널 수 (예제: 640x480 RGB)
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
CHANNELS = 3
TOTAL_FRAMES = 27
TARGET_FRAME = 9  # 0-based index, 10번째 프레임은 index 9

def load_rgb_frame(bin_path, frame_idx):
    frame_size = IMAGE_HEIGHT * IMAGE_WIDTH * CHANNELS  # 한 프레임의 데이터 크기
    total_size = frame_size * TOTAL_FRAMES  # 전체 데이터 크기

    # binary 파일 읽기
    with open(bin_path, "rb") as f:
        raw_data = f.read(total_size)  # 전체 데이터 로드

    # numpy 배열로 변환
    img_array = np.frombuffer(raw_data, dtype=np.uint8)

    # 특정 프레임 추출
    start = frame_idx * frame_size
    end = start + frame_size
    frame = img_array[start:end]

    # 이미지 reshape (가정: RGB 순서, 높이 x 너비 x 채널)
    frame = frame.reshape((IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
    
    return frame

def save_image(image, save_path):
    cv2.imwrite(save_path, image)

if __name__ == "__main__":
    rgb_image = load_rgb_frame(Config.rgb_path, TARGET_FRAME)
    save_path = os.path.join(Config.current_path, "frame_10.png")
    save_image(rgb_image, save_path)
    print(f"10th frame saved at: {save_path}")

