import pandas as pd
from numpy.linalg import inv
import numpy as np
import math
import cv2
import torch
import os

from matplotlib import pyplot as plt

model = torch.hub.load('/home/wql/yolov5', 'custom', path='best.pt', source='local')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

class KalmanFilter():
    def __init__(self,
                 xinit: int = 0,
                 yinit: int = 0,
                 fps: int = 30,
                 std_a: float = 0.001,
                 std_x: float = 0.0045,
                 std_y: float = 0.01,
                 cov: float = 100000) -> None:

        # State Matrix
        self.S = np.array([xinit, 0, 0, yinit, 0, 0])
        self.dt = 1 / fps

        # State Transition Model
        # Here, we assume that the model follow Newtonian Kinematics
        self.F = np.array([[1, self.dt, 0.5 * (self.dt * self.dt), 0, 0, 0],
                           [0, 1, self.dt, 0, 0, 0], [0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 1, self.dt, 0.5 * self.dt * self.dt],
                           [0, 0, 0, 0, 1, self.dt], [0, 0, 0, 0, 0, 1]])

        self.std_a = std_a

        # Process Noise
        self.Q = np.array([
            [
                0.25 * self.dt * self.dt * self.dt * self.dt, 0.5 * self.dt *
                self.dt * self.dt, 0.5 * self.dt * self.dt, 0, 0, 0
            ],
            [
                0.5 * self.dt * self.dt * self.dt, self.dt * self.dt, self.dt,
                0, 0, 0
            ], [0.5 * self.dt * self.dt, self.dt, 1, 0, 0, 0],
            [
                0, 0, 0, 0.25 * self.dt * self.dt * self.dt * self.dt,
                0.5 * self.dt * self.dt * self.dt, 0.5 * self.dt * self.dt
            ],
            [
                0, 0, 0, 0.5 * self.dt * self.dt * self.dt, self.dt * self.dt,
                self.dt
            ], [0, 0, 0, 0.5 * self.dt * self.dt, self.dt, 1]
        ]) * self.std_a * self.std_a

        self.std_x = std_x
        self.std_y = std_y

        # Measurement Noise
        self.R = np.array([[self.std_x * self.std_x, 0],
                           [0, self.std_y * self.std_y]])

        self.cov = cov

        # Estimate Uncertainity
        self.P = np.array([[self.cov, 0, 0, 0, 0, 0],
                           [0, self.cov, 0, 0, 0, 0],
                           [0, 0, self.cov, 0, 0, 0],
                           [0, 0, 0, self.cov, 0, 0],
                           [0, 0, 0, 0, self.cov, 0],
                           [0, 0, 0, 0, 0, self.cov]])

        # Observation Matrix
        # Here, we are observing X & Y (0th index and 3rd Index)
        self.H = np.array([[1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]])

        self.I = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])

        # Predicting the next state and estimate uncertainity
        self.S_pred = None
        self.P_pred = None

        # Kalman Gain
        self.K = None

        # Storing all the State, Kalman Gain and Estimate Uncertainity
        self.S_hist = [self.S]
        self.K_hist = []
        self.P_hist = [self.P]

    def pred_new_state(self):
        self.S_pred = self.F.dot(self.S)

    def pred_next_uncertainity(self):
        self.P_pred = self.F.dot(self.P).dot(self.F.T) + self.Q

    def get_Kalman_gain(self):
        self.K = self.P_pred.dot(self.H.T).dot(
            inv(self.H.dot(self.P_pred).dot(self.H.T) + self.R))
        self.K_hist.append(self.K)

    def state_correction(self, z):
        if z == [None, None]:
            self.S = self.S_pred
        else:
            self.S = self.S_pred + +self.K.dot(z - self.H.dot(self.S_pred))

        self.S_hist.append(self.S)

    def uncertainity_correction(self, z):
        if z != [None, None]:
            self.l1 = self.I - self.K.dot(self.H)
            self.P = self.l1.dot(self.P_pred).dot(self.l1.T) + self.K.dot(
                self.R).dot(self.K.T)
        self.P_hist.append(self.P)

def draw_prediction(img: np.ndarray,
                    class_name: str,
                    df: pd.core.series.Series,
                    color: tuple = (255, 0, 0)):
    '''
    Function to draw prediction around the bounding box identified by the YOLO
    The Function also displays the confidence score top of the bounding box 
    '''

    cv2.rectangle(img, (int(df.xmin), int(df.ymin)),
                  (int(df.xmax), int(df.ymax)), color, 2)
    cv2.putText(img, class_name + " " + str(round(df.confidence, 2)),
                (int(df.xmin) - 10, int(df.ymin) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img

def convert_video_to_frame(path: str):
    '''
    The function take input as video file and returns a list of images for every video
    '''

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    img = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            img.append(frame)
        else:
            break

    cap.release()
    return img, fps

def record_video_and_save_frames(output_video_path: str, duration: int):
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("无法打开摄像头！")

    # 获取摄像头的帧率和分辨率
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 设置视频编码器和输出
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 存储帧的列表
    frames = []

    print(f"开始录制视频，持续 {duration} 秒。按 'q' 提前退出。")

    frame_count = 0
    while frame_count < duration * fps:
        ret, frame = cap.read()
        if not ret:
            print("无法读取帧，可能是摄像头问题。")
            break

        # 显示当前帧
        cv2.imshow('Recording', frame)

        # 保存帧到视频文件
        out.write(frame)

        # 保存帧到帧列表
        frames.append(frame)

        # 检测是否按下 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"视频录制完成，已保存到 {output_video_path}")
    return frames, fps


# 调用函数录制视频并保存帧
frames, fps = convert_video_to_frame('output.avi')  # 录制 5 秒视频


results = model(frames)

df_handle = results.pandas().xyxy

filter_handle = KalmanFilter(fps=fps, xinit=60,
                          yinit=150, std_x=0.000025, std_y=0.0001)

for df in df_handle:
    df = df.loc[df['name'] == 'handler']
    x_cen, y_cen = None, None

    if len(df) > 0:
        x_cen = (df.xmin.values[0] + df.xmax.values[0]) / 2
        y_cen = (df.ymin.values[0] + df.ymax.values[0]) / 2

    filter_handle.pred_new_state()
    filter_handle.pred_next_uncertainity()
    filter_handle.get_Kalman_gain()
    filter_handle.state_correction([x_cen, y_cen])
    filter_handle.uncertainity_correction([x_cen, y_cen])

# out = cv2.VideoWriter('handle_kalman.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10,
#                       (frames[0].shape[1], frames[0].shape[0]))
output_folder = "processed_frames"

for i in range(len(frames)):
    x, y = filter_handle.S_hist[i][0], filter_handle.S_hist[i][3]
    df = df_handle[i].loc[df_handle[i]['name'] == 'handler']
    tmp_img = frames[i]

    for j in df.index.values:
        tmp_img = draw_prediction(tmp_img, 'handler', df.loc[j])

    tmp_img = cv2.circle(tmp_img, (math.floor(
        filter_handle.S_hist[i][0]), math.floor(filter_handle.S_hist[i][3])),
        radius=1,
        color=(255, 0, 0),
        thickness=10)

    output_path = os.path.join(output_folder, f"frame_{i:04d}.png")
    cv2.imwrite(output_path, tmp_img)

