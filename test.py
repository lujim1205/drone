import torch
import cv2

# 1. 載入模型
model = torch.hub.load(
    'ultralytics/yolov5', 
    'yolov5s', 
    device='cuda:0' if torch.cuda.is_available() else 'cpu',
    trust_repo=True
)

# 2. 讀取剛剛放進來的 test.jpg
img_path = 'test.jpg'
img = cv2.imread(img_path)

if img is None:
    print("❌ 找不到 test.jpg，請確認照片有放在 C:\\projects 資料夾裡！")
else:
    # 3. 讓 YOLOv5 進行辨識
    results = model(img)
    results.print()  # 印出辨識到的物件

    # 4. 畫出框框並跳出視窗顯示
    rendered_img = results.render()[0]
    cv2.imshow('YOLOv5 Test', rendered_img)
    
    print("按下任意鍵或關閉視窗即可結束...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()