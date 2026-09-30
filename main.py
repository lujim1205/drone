import os
import sys
import time
import signal
import logging
import warnings
from datetime import datetime
from threading import Thread, Lock
import cv2
import numpy as np
import torch
import keyboard
from djitellopy.tello import Tello
from simple_pid import PID

# 1. YOLOv5
YOLO_DIR = r'C:\Users\yalin\.cache\torch\hub\ultralytics_yolov5_master'

GLOBAL_MODEL = torch.hub.load(
    YOLO_DIR, 
    'custom', 
    path='yolov5s.pt', 
    source='local', 
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
)

class TelloDrone(Tello):
    def __init__(self):
        super().__init__()
        self.connect()
        print("battery: ", self.get_battery())
        
        self.model = GLOBAL_MODEL
        self.lock_class_id = None
        self.last_locked_class_id = None  # 記錄上一次鎖定的類別
        self.detections = []
        
        # 距離過濾器相關
        self.last_target_pos = None
        self.max_tracking_distance = 180

        # 控制變數與程式狀態旗標
        self.running = True
        self.limit = 70
        self.frame_center = (480, 360)
        self.lr, self.fb, self.ud, self.yv = 0, 0, 0, 0
        self.cmd_lock = Lock()
        
        self.space_state = 0
        self.rev_on = False
        self.rev_speed = 0
        self.manual_flag = False
        self.update_width_flag = False
        self.target_width = 0

        # PID 設定
        self.pid_fb = PID(1.4, 0.05, 0.3, setpoint=0)
        self.pid_ud = PID(0.145, 0.001, 0.06, setpoint=0)
        self.pid_yv = PID(0.127, 0.005, 0.06, setpoint=0)

        # Log 資料夾
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(current_time, exist_ok=True)
        print(current_time)
        
        self.setting_file_path = os.path.join(current_time, "settings.txt")
        self.log_file_path = os.path.join(current_time, "log.txt")
        
        with open(self.setting_file_path, "w") as log_file:
            log_file.write(f"{self.pid_fb}\n{self.pid_ud}\n{self.pid_yv}\n{self.limit}\n")

        # 啟動鍵盤執行緒 (Daemon)
        self.keyboard_thread = Thread(target=self.getKeyboardInput, daemon=True)
        self.keyboard_thread.start()

        # 啟動串流 (加入防衝擊等待與重試機制)
        print("drone video starting...")
        self.cap = None
        for attempt in range(3):
            try:
                self.streamoff()
                time.sleep(1.0)
                self.streamon()
                time.sleep(2.0)  # 等待 H.264 關鍵影格
                self.cap = self.get_frame_read()
                time.sleep(1.0)
                if self.cap.frame is not None:
                    print("[INFO] 影像串流連線成功！")
                    break
            except Exception as e:
                print(f"[WARNING] 串流啟動失敗，第 {attempt + 1} 次重試... (Error: {e})")
                time.sleep(1.5)

        if self.cap is None:
            raise RuntimeError("無法從 Tello 獲取影片串流，請檢查 Wi-Fi 連線與防火牆設定。")
    def stop_all(self):
        """安全停止全域狀態"""
        if self.running:
            print("\n[INFO] 正在安全關閉無人機系統...")
            self.running = False

    def relock_target(self):
        """按下 Space 重新鎖定當前畫面目標並恢復公轉"""
        if len(self.detections) == 0:
            print("[RELOCK FAIL] 畫面上沒有偵測到任何目標")
            return

        target_det = None
        # 若之前有鎖定過類別，優先鎖定該類別
        if self.last_locked_class_id is not None:
            candidates = [d for d in self.detections if d[5] == self.last_locked_class_id]
            if candidates:
                target_det = min(candidates, key=lambda d: np.hypot(d[0] - self.frame_center[0], d[1] - self.frame_center[1]))
        
        if target_det is None:
            target_det = min(self.detections, key=lambda d: np.hypot(d[0] - self.frame_center[0], d[1] - self.frame_center[1]))

        x_center, y_center, width, height, confidence, class_id = target_det
        self.lock_class_id = class_id
        self.last_locked_class_id = class_id
        self.target_width = width
        self.last_target_pos = (x_center, y_center)

        # 重置 PID 並開啟公轉
        self.pid_fb.reset()
        self.pid_ud.reset()
        self.pid_yv.reset()
        self.rev_on = True
        self.rev_speed = 0
        print(f"[RELOCK SUCCESS] 重新鎖定類別: {class_id}, 基準寬度: {width:.2f}, 公轉開啟 (Speed: {self.rev_speed})")

    def getKeyboardInput(self):
        step = 5
        speed = 50
        hold1 = 0
        hold2 = 0
        
        while self.running:
            lr, fb, ud, yv = 0, 0, 0, 0
            
            if keyboard.is_pressed("LEFT"): lr = -speed
            elif keyboard.is_pressed("RIGHT"): lr = speed

            if keyboard.is_pressed("UP"): 
                fb = speed
                self.update_width_flag = True 
            elif keyboard.is_pressed("DOWN"): 
                fb = -speed
                self.update_width_flag = True
            else:
                self.update_width_flag = False
            
            if keyboard.is_pressed("w"): ud = speed
            elif keyboard.is_pressed("s"): ud = -speed

            if keyboard.is_pressed("a"): yv = -speed
            elif keyboard.is_pressed("d"): yv = speed

            # 9 鍵安全降落
            if keyboard.is_pressed("9"): 
                print("[KEYBOARD] 偵測到 9 鍵，準備降落...")
                self.stop_all()
                break
            
            if keyboard.is_pressed("e"): 
                self.takeoff()

            if keyboard.is_pressed("1") and self.rev_speed > -100:
                if hold1 == 0:
                    self.rev_speed -= step
                    hold1 = 1
                    print("revolution speed: ", self.rev_speed)
            else: 
                hold1 = 0
            
            if keyboard.is_pressed("2") and self.rev_speed < 100:
                if hold2 == 0:
                    self.rev_speed += step
                    hold2 = 1
                    print("revolution speed: ", self.rev_speed)
            else: 
                hold2 = 0

            # 空白鍵 (Space) 重新鎖定並啟動公轉
            if keyboard.is_pressed("space") and self.space_state == 0:
                self.space_state = 1
            if not keyboard.is_pressed("space") and self.space_state == 1:
                self.space_state = 0
                self.relock_target()

            # WASD / 方向鍵手動接管邏輯
            has_manual_input = any(v != 0 for v in [lr, fb, ud, yv])
            
            with self.cmd_lock:
                if has_manual_input:
                    self.manual_flag = True
                    if self.lock_class_id is not None or self.rev_on:
                        self.lock_class_id = None
                        self.rev_on = False
                        print("[MANUAL OVERRIDE] 偵測到手動輸入，已關閉公轉與自動鎖定！")
                    
                    self.lr, self.fb, self.ud, self.yv = lr, fb, ud, yv
                else:
                    self.manual_flag = False
                    if self.lock_class_id is None:
                        self.lr = self.rev_speed if self.rev_on else 0
                        self.fb, self.ud, self.yv = 0, 0, 0

            time.sleep(0.03)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Left button clicked at ({x}, {y})")
            for det in self.detections:
                x_center, y_center, width, height, confidence, class_id = det
                if abs(x - x_center) < width / 2 and abs(y - y_center) < height / 2:
                    self.lock_class_id = class_id
                    self.last_locked_class_id = class_id
                    self.target_width = width
                    self.last_target_pos = (x_center, y_center)
                    
                    self.pid_fb.reset()
                    self.pid_ud.reset()
                    self.pid_yv.reset()
                    print(f"[LOCK] 滑鼠鎖定類別: {class_id}, 基準寬度設為: {self.target_width:.2f}")
                    break

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.lock_class_id = None
            self.last_target_pos = None
            self.rev_on = False
            print(f"Right button clicked: 取消鎖定與公轉")

    def adj_pose(self, ud_dif, fb_dif, yv_dif):
        lr = self.rev_speed if self.rev_on else 0
        
        if abs(fb_dif) < 8:
            fb = 0
        else:
            fb = int(self.pid_fb(fb_dif))

        ud = int(self.pid_ud(ud_dif))
        yv = -int(self.pid_yv(yv_dif))

        self.lr = max(-self.limit, min(self.limit, lr))
        self.fb = max(-self.limit, min(self.limit, fb))
        self.ud = max(-self.limit, min(self.limit, ud))
        self.yv = max(-self.limit, min(self.limit, yv))

    def drone_frame(self):
        cv2.namedWindow('Detection Result')
        cv2.setMouseCallback('Detection Result', self.mouse_callback)

        periods = []
        period_window_size = 10
        prev_time = time.time()
        fps = 0

        try:
            while self.running:
                # 關鍵修復：加入安全讀影格與防護機制
                try:
                    frame = self.cap.frame
                except Exception as e:
                    time.sleep(0.01)
                    continue

                if frame is None or frame.size == 0:
                    time.sleep(0.01)
                    continue

                image = frame.copy()
                results = self.model(image)
                
                curr_time = time.time()
                periods.append(round(curr_time - prev_time, 3)) 
                prev_time = curr_time
                if len(periods) == period_window_size: 
                    fps = 1 / (sum(periods) / len(periods))
                    periods.pop(0)

                self.detections = results.xywh[0].cpu().numpy()

                # 歐氏距離過濾
                best_target = None
                min_dist = float('inf')

                if self.lock_class_id is not None and self.last_target_pos is not None:
                    for det in self.detections:
                        x_center, y_center, width, height, confidence, class_id = det
                        if class_id == self.lock_class_id:
                            dist = np.hypot(x_center - self.last_target_pos[0], y_center - self.last_target_pos[1])
                            if dist < min_dist and dist < self.max_tracking_distance:
                                min_dist = dist
                                best_target = det

                if self.lock_class_id is None and not self.manual_flag:
                    with self.cmd_lock:
                        self.lr = self.rev_speed if self.rev_on else 0
                        self.fb, self.ud, self.yv = 0, 0, 0

                for det in self.detections:
                    x_center, y_center, width, height, confidence, class_id = det
                    x1 = int(x_center - width / 2)
                    y1 = int(y_center - height / 2)
                    x2 = int(x_center + width / 2)
                    y2 = int(y_center + height / 2)
                    label = f"{results.names[int(class_id)]} {confidence:.2f}"

                    is_locked = (best_target is not None) and np.array_equal(det, best_target)

                    if is_locked:
                        self.last_target_pos = (x_center, y_center)
                        
                        ud_dif = y_center - self.frame_center[1]
                        fb_dif = width - self.target_width
                        yv_dif = x_center - self.frame_center[0]
                        
                        with self.cmd_lock:
                            if not self.manual_flag:
                                self.adj_pose(ud_dif, fb_dif, yv_dif)

                        # UI 指示
                        circle_color = (0, 255, 255) if self.fb > 0 else (0, 0, 255)
                        cv2.circle(image, self.frame_center, abs(self.fb), circle_color, 2)
                        cv2.arrowedLine(image, self.frame_center, (self.frame_center[0], self.frame_center[1] - self.ud), (0, 255, 255), 2)
                        cv2.arrowedLine(image, self.frame_center, (self.frame_center[0] + self.yv, self.frame_center[1]), (0, 255, 255), 2)

                        cv2.circle(image, (int(x_center), int(y_center)), 2, (255, 0, 0), 2)
                        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                        bounding_box = [round(float(x_center), 3), round(float(y_center), 3), round(float(width), 3), round(float(height), 3)]
                        differences = [round(float(ud_dif), 3), round(float(fb_dif), 3), round(float(yv_dif), 3)]
                        
                        with open(self.log_file_path, "a") as log_file:
                            log_file.write(f"Time: {curr_time};bat: {self.get_battery()};{bounding_box};{differences};{[self.lr, self.fb, self.ud, self.yv]}\n")
                    else:
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # 發送指令
                with self.cmd_lock:
                    self.send_rc_control(self.lr, self.fb, self.ud, self.yv)

                cv2.circle(image, self.frame_center, 2, (255, 255, 255), 2)
                cv2.putText(image, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Detection Result', image)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('9') or key == 27:
                    print("[CV2] 偵測到退出按鍵...")
                    self.stop_all()
                    break

        finally:
            print("[CLEANUP] 開始執行資源釋放與安全降落...")
            try:
                self.send_rc_control(0, 0, 0, 0)
                self.land()
            except Exception as e:
                print(f"[CLEANUP ERROR] Land failed: {e}")

            try:
                self.streamoff()
            except Exception as e:
                print(f"[CLEANUP ERROR] Streamoff failed: {e}")

            cv2.destroyAllWindows()
            try:
                self.end()
            except Exception as e:
                print(f"[CLEANUP ERROR] Tello end failed: {e}")
                
            print("[CLEANUP] 程式安全結束。")

if __name__ == '__main__':
    drone = TelloDrone()

    def signal_handler(sig, frame):
        print("\n[SIGNAL] 偵測到 Ctrl+C (SIGINT)，強制安全關閉中...")
        drone.stop_all()

    signal.signal(signal.SIGINT, signal_handler)

    drone.drone_frame()