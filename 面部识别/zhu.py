import dlib
import cv2
import matplotlib.pyplot as plt

# 加载dlib的人脸检测器和特征提取器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 读取人脸图像
image = cv2.imread("your_face_image.jpg")

# 转换图像为灰度
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 使用dlib的人脸检测器检测人脸
faces = detector(gray)

# 初始化一个空的特征点列表
landmarks_list = []

# 遍历检测到的每张人脸
for face in faces:
    # 使用dlib的特征提取器获取特征点
    landmarks = predictor(gray, face)
    
    # 将特征点转换为坐标
    landmarks_points = []
    for n in range(68):  # 68个特征点
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmarks_points.append((x, y))
    
    # 将特征点添加到特征点列表
    landmarks_list.append(landmarks_points)

# 绘制散点图
for landmarks_points in landmarks_list:
    x = [point[0] for point in landmarks_points]
    y = [point[1] for point in landmarks_points]
    plt.scatter(x, y, c='k')

# 显示图像和散点图
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()