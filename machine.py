import cv2

# 初始化摄像头
cap = cv2.VideoCapture(0)
# # 读取视频
# cap = cv2.VideoCapture('D:/everything/mc.mp4')

# 读取第一帧并选择模板
ret, frame = cap.read()
template = cv2.selectROI("Select Template", frame, fromCenter=False)
template_img = frame[int(template[1]):int(template[1] + template[3]), int(template[0]):int(template[0] + template[2])]
h, w = template_img.shape[:2]

# 开始跟踪
while True:
    _, frame = cap.read()
    if not ret:
        break

    # 匹配模板
    res = cv2.matchTemplate(frame, template_img, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(res)

    # 绘制跟踪结果
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()