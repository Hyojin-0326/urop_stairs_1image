import utils
import yolo
import time
import os

##### 데이터 로드,경로 또 꼬이면 경로 yolo.config에서 수정해야 함
class Config:
    current_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_path, "yolo", "best.pt")
    meta_path = os.path.join(current_path, "data", "meta.txt")
    rgb_path = os.path.join(current_path, "data", "rgb_data.bin")
    depth_path = os.path.join(current_path, "data", "depth_data.bin")

depth_map, rgb_image = utils.align_depth_to_rgb(Config.depth_path, Config.rgb_path, 10)

##### YOLO 모델 로드
results = yolo.detect(rgb_image)

#전처리 위한 pointcloud 변환 -> groundplane 날리기 -> rgb img에서도 groundplane 자르기
points = utils.depth_to_pointcloud(depth_map) # 3d 좌표로 반환됨

print(type(points))  # numpy.ndarray 확인
print(points.shape)  # 형태인지 확인


#디버깅용
utils.pointcloud_visualization(points)

# points_without_ground, ground_points = utils.preprocessPointCloud(points)

# rgb_without_ground = utils.preprocess_RGBimg(rgb_image, points, ground_points)




# ###### 만약 바운딩박스가 여러 개면 가까운거 1개만 남기고 없애기
# has_duplicate = utils.check_duplicate(results)
# if has_duplicate:
#     cls_id, bbox = utils.remove_extra_box(results, depth_map)
# else:
#     cls_id = int(results[0].boxes.cls[0])
#     bbox = tuple(map(int, results[0].boxes.xyxy[0]))

# yolo.draw_and_save_final_bbox(rgb_image, bbox)



# ###### 연산량 감소를 위해 ROI 크롭
# rgb_roi, depth_roi = utils.crop_roi(bbox, rgb_image, depth_map)

# ###### ROI 내에서 탐지된 물체의 height 구하기 (5프레임동안 모아서 평균)
# height = utils.measure_height(cls_id, rgb_roi, depth_roi, bbox)




# end_time = time.time()
# exe_time = end_time - start_time
# print(f'⏳ {exe_time:.4f}초 동안 실행됨')
