import cv2
import numpy as np
import supervision as sv
from PIL import Image
from pathlib import Path
from rfdetr import RFDETRSegSmall

# 1. Khai báo đường dẫn file
BASE_DIR = Path(__file__).resolve().parent
model_path = BASE_DIR / "models" / "RF-DETR_Small.pt"
image_path = BASE_DIR / "image.png"

def test_rfdetr_segmentation():
    try:
        print("Đang tải mô hình RF-DETR Segmentation (Small)...")
        
        # Load mô hình với file weights (.pt) của bạn
        # LƯU Ý QUAN TRỌNG: Nếu mô hình này bạn train trên dữ liệu riêng (không phải 80 class mặc định của COCO),
        # bạn CẦN truyền thêm tham số num_classes (ví dụ: mô hình có 5 nhãn thì thêm num_classes=5)
        # model = RFDETRSegSmall(pretrain_weights=model_path, num_classes=5)
        model = RFDETRSegSmall(pretrain_weights=model_path)
        
        # Đọc ảnh bằng PIL để chuẩn hệ màu RGB (yêu cầu của rfdetr)
        image = Image.open(image_path).convert("RGB")
        
        print("Đang tiến hành phân vùng (Segmentation)...")
        # Chạy dự đoán với ngưỡng tin cậy 0.5 (có thể tuỳ chỉnh)
        detections = model.predict(image, threshold=0.5)
        
        # Chuyển đổi ảnh sang định dạng Numpy/OpenCV (BGR) để vẽ và hiển thị
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # 2. Sử dụng thư viện Supervision để vẽ Mask và Bounding Box
        mask_annotator = sv.MaskAnnotator()
        box_annotator = sv.BoxAnnotator()
        
        # Phủ lớp mask (vùng pixel) và vẽ viền hộp lên ảnh
        annotated_image = mask_annotator.annotate(scene=image_cv.copy(), detections=detections)
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
        
        # Hiển thị kết quả bằng OpenCV
        cv2.imshow("Ket qua RF-DETR Instance Segmentation", annotated_image)
        print("Hoàn tất! Nhấn phím bất kỳ trên cửa sổ ảnh để đóng...")
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Đã có lỗi xảy ra: {e}")

if __name__ == "__main__":
    test_rfdetr_segmentation()
