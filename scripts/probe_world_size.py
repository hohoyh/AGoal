import airsim
import time
import os

# --- 1. 连接、起飞 (和之前一样) ---
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
# ==========================================================
# ！！在这里加入新代码！！
print("Disabling auto-exposure by locking exposure...")
client.simSetCameraPostProcessSettings(
    camera_name="0", # 对应你 settings.json 里的相机 ID
    post_process_settings=airsim.PostProcessSettings(
        auto_exposure_method=airsim.PostProcessSettings.AutoExposureMethod.Histogram, # 使用直方图方法
        auto_exposure_min_brightness=1.0, # 锁定最小亮度
        auto_exposure_max_brightness=1.0, # 锁定最大亮度
        auto_exposure_speed=100
    ),
    vehicle_name="Drone1" # 对应你的无人机名字
)
print("Auto-exposure disabled.")
# ==========================================================
print("Taking off...")
client.takeoffAsync().join()
print("Takeoff complete.")

# --- 2. 飞到一个新位置 ---
# 移动到前方10米, 离地5米高 (NED 坐标系: X=前, Y=右, Z=下)
# 注意：Z=-5 表示在空中5米
print("Flying to new position (X=10, Y=0, Z=-5)...")
client.moveToPositionAsync(
     2,  # X (m)
    -10,   # Y (m)
    -5,  # Z (m)
    5    # 速度 (m/s)
).join()
print("Move complete.")

# --- 3. 定义你想要的图像 ---
# 我们根据 settings.json 请求 "0" 号相机的 RGB (Scene) 图像
image_request = airsim.ImageRequest(
    "0",                       # Camera name, 对应 settings.json 里的 "0"
    airsim.ImageType.Scene,    # ImageType 0 (RGB)
    pixels_as_float=False,     # False: 返回 8-bit (uint8) PNG 压缩数据
    compress=True              # True: 压缩为 PNG (更快)
)

print("Requesting image...")

# --- 4. 获取图像数据 ---
# simGetImages 可以一次接收一个列表的请求
responses = client.simGetImages([image_request])
response = responses[0]

# --- 5. 保存图像到文件 ---
if response.image_data_uint8:
    # 确保保存图像的目录存在
    if not os.path.exists('airsim_captures'):
        os.makedirs('airsim_captures')

    # 将 PNG 字节数据写入文件
    filename = 'airsim_captures/capture_01.png'
    with open(filename, 'wb') as f:
        f.write(response.image_data_uint8)
    
    print(f"Image saved to: {os.path.abspath(filename)}")
else:
    print("Error: Image data was empty.")

# --- 6. 任务完成，安全降落 ---
print("Landing...")
client.landAsync().join()
print("Landed.")

client.armDisarm(False)
client.enableApiControl(False)
print("Script complete.")