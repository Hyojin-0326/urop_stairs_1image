import utils
import yolo
import time
import os
import stairs
import numpy as np
import matplotlib.pyplot as plt

##### 데이터 로드,경로 또 꼬이면 경로 yolo.config에서 수정해야 함
class Config:
    current_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_path, "yolo", "best.pt")
    meta_path = os.path.join(current_path, "data", "meta.txt")
    rgb_path = os.path.join(current_path, "data", "rgb_data.bin")
    depth_path = os.path.join(current_path, "data", "depth_data.bin")

depth_map, rgb_image = utils.align_depth_to_rgb(Config.depth_path, Config.rgb_path, 10)

# ##### YOLO 모델 로드
results = yolo.detect(rgb_image)


#----------------해야되나? groundplane 날리는거 < 근데 얘가 오히려 더 연산량 잡아먹을수도
#전처리 위한 pointcloud 변환 -> groundplane 날리기 -> rgb img에서도 groundplane 자르기

# print(type(points))  # numpy.ndarray 확인, list임

#디버깅용
#utils.pointcloud_visualization(points)


# points_without_ground, ground_points = utils.preprocessPointCloud(points)

# rgb_without_ground = utils.preprocess_RGBimg(rgb_image, points, ground_points)

# results = yolo.detect(rgb_without_ground)
# yolo.draw_and_save_bbox(rgb_without_ground, results) #for debug



###### 만약 바운딩박스가 여러 개면 가까운거 1개만 남기고 없애기
has_duplicate = utils.check_duplicate(results)
if has_duplicate:
    cls_id, bbox = utils.remove_extra_box(results, depth_map)
else:
    cls_id = int(results[0].boxes.cls[0])
    bbox = tuple(map(int, results[0].boxes.xyxy[0]))

yolo.draw_and_save_final_bbox(rgb_image, bbox)



###### 연산량 감소를 위해 ROI 크롭
rgb_roi, depth_roi = utils.crop_roi(bbox, rgb_image, depth_map)
#plt.imsave('saved_image.png', rgb_roi) 디버깅용


###### ROI 내에서 탐지된 물체의 height 구하기
roi_points = utils.depth_to_pointcloud(depth_roi) # depth_roi가 3d 좌표로 반환됨
height = stairs.measure_height(roi_points)

closest_plane, closest_normal, inlier_points = utils.extract_plane_ransac(roi_points) # roi 내에서 평면 찾음



#디버깅
def debug_projection_rgb(rgb_image, points):
    """
    3D 포인트 클라우드를 RGB 이미지 상의 좌표로 투영하고, 해당 RGB 값 추출.
    
    :param depth_image: Depth 이미지 (H, W)
    :param rgb_image: RGB 이미지 (H, W, 3)
    :param points: 3D 포인트 (N x 3) numpy 배열
    :param intrinsics: 카메라 내재 파라미터 (fx, fy, cx, cy)
    :return: RGB 값들
    """
    # 3D 포인트를 2D 이미지 좌표로 변환
    intrinsics= 605.9815, 606.1337, 308.5001, 246.4238
    uv_coords = utils.project_to_2d(points, intrinsics=intrinsics)

    # 2D 좌표를 정수형으로 반올림하여 인덱싱
    u = np.round(uv_coords[:, 0]).astype(int)
    v = np.round(uv_coords[:, 1]).astype(int)

    # 이미지 크기를 넘어가는 좌표는 무시
    h, w, _ = rgb_image.shape
    u = np.clip(u, 0, w - 1)
    v = np.clip(v, 0, h - 1)

    # RGB 값 추출
    rgb_values = rgb_image[v, u]

    # RGB 이미지를 디버깅용으로 출력
    debug_image = rgb_image.copy()
    for (x, y), rgb in zip(zip(u, v), rgb_values):
        debug_image[y, x] = [255, 0, 0]  # 빨간 점으로 표시

    plt.imsave('plane_rgb_image.png', debug_image)

    return rgb_values, debug_image



debug_projection_rgb(rgb_roi, inlier_points) #inlier points부분 이미지 저장






