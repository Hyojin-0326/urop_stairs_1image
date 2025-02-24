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
