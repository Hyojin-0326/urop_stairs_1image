import cv2
import torch
import numpy as np
from ultralytics import YOLO
import utils
import matplotlib.pyplot as plt
import os



#---------------전역변수모음
class Config:
    current_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_path, "yolo", "best.pt")
    meta_path = os.path.join(current_path, "data", "meta.txt")
    rgb_path = os.path.join(current_path, "data", "rgb_data.bin")
    depth_path = os.path.join(current_path, "data", "depth_data.bin")

    # trt_path=os.path.join(current_path, "yolo", "yolo_model.trt")
    # engine_path = os.path.join(current_path, "yolo", "yolo_model.trt")
    # onnx_path=os.path.join(current_path, "yolo", "yolo_model.onnx")
    


def detect(rgb_image):
    model = YOLO(Config.model_path)
    resized_image = cv2.resize(rgb_image, (640, 480))

    # 2. 이미지를 RGB 형식으로 변환 (이미지 로딩 시 BGR일 수 있음)
    #rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

    # 3. 이미지를 텐서로 변환하고, 정규화 (0~255 -> 0~1)
    tensor_image = torch.from_numpy(rgb_image).float()  # numpy 배열을 텐서로 변환
    tensor_image /= 255.0  # 0~1로 정규화
    tensor_image = tensor_image.permute(2, 0, 1).unsqueeze(0)  # (H, W, C) -> (1, C, H, W) 형태로 변환

    # 4. 모델로 객체 감지 수행
    results = model(tensor_image)  # 감지 결과 반환

    return results





#--------걍 테스트용-----------



def main():
    model_path = "best.pt"
    image_path = "test.jpg"

    model = YOLO(model_path)  # YOLO 모델 로드
    results = model(image_path)  # 객체 탐지 실행



    print(results)  # 결과 출력 (디버깅용)

if __name__ == "__main__":
    main()

# -----창고---------

def draw_and_save_bbox(rgb_image, results, save_path="detected_result.png"):
    """
    YOLO 결과를 기반으로 바운딩 박스를 그려서 저장하는 함수

    Args:
        rgb_image (numpy.ndarray): YOLO에 입력한 원본 이미지 (numpy 배열)
        results (list): YOLO 감지 결과 리스트
        save_path (str): 저장할 이미지 경로
    """

    # Matplotlib에서 RGB 순서로 보여주기 위해 변환 (cv2는 BGR 기본)
    image_with_boxes = rgb_image.copy()

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # 바운딩 박스 좌표
            cls_id = int(box.cls[0])  # 감지된 클래스 ID
            conf = box.conf[0]  # 신뢰도

            # 바운딩 박스 그리기
            cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Class {cls_id}: {conf:.2f}"
            cv2.putText(image_with_boxes, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 255, 0), 2)

    # Matplotlib을 사용하여 저장
    plt.figure(figsize=(8, 6))
    plt.imshow(image_with_boxes)
    plt.axis("off")  # 축 제거
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    print(f"✅ 바운딩 박스가 포함된 이미지 저장 완료: {save_path}")

def draw_and_save_final_bbox(rgb_image, bbox, save_path="final_bbox.png"):
    """
    최종 바운딩 박스를 이미지에 그리고 저장하는 함수

    Args:
        rgb_image (numpy.ndarray): 원본 RGB 이미지
        bbox (tuple): 최종 남은 바운딩 박스 (x1, y1, x2, y2)
        save_path (str): 저장할 이미지 경로
    """
    x1, y1, x2, y2 = bbox  # 바운딩박스 좌표 가져오기

    # 바운딩 박스 그리기
    image_with_box = rgb_image.copy()
    cv2.rectangle(image_with_box, (x1, y1), (x2, y2), (0, 255, 0), 3)  # 초록색 박스
    label = f"Class {cls_id}"
    cv2.putText(image_with_box, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Matplotlib을 사용해 저장
    plt.figure(figsize=(8, 6))
    plt.imshow(image_with_box)
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    print(f"✅ 최종 바운딩박스 포함된 이미지 저장 완료: {save_path}")
