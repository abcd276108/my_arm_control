import cv2
import socket
import struct
import numpy as np
import os

save_dir = "dataset/images"
os.makedirs(save_dir, exist_ok=True)
count = 0

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('192.168.51.188', 9999)) # 這裡就是從筆電連向手臂的關鍵！
print("✅ 成功連上手臂鏡頭！")

data = b""
payload_size = struct.calcsize("<L")

while True:
    while len(data) < payload_size:
        packet = client_socket.recv(4096)
        if not packet: break
        data += packet
    if not data: break

    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack("<L", packed_msg_size)[0]

    while len(data) < msg_size:
        data += client_socket.recv(4096)

    frame_data = data[:msg_size]
    data = data[msg_size:]

    frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
    if frame is not None:
        cv2.imshow("Laptop AI Vision (Press 's' to capture, 'q' to quit)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            img_name = os.path.join(save_dir, f"sample_{count:03d}.jpg")
            cv2.imwrite(img_name, frame)
            print(f"📸 拍照成功: {img_name}")
            count += 1
        elif key == ord('q'):
            break