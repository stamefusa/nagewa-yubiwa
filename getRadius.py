import cv2
import numpy as np

def img2bin(img):    
    """画像の二値化

    Args:
        img: 元画像
    Returns:
        Any: 二値化画像
    """
    # 閾値の設定
    threshold = 50

    # 二値化(閾値100を超えた画素を255にする。)
    ret, img_thresh = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

    # オープニング処理
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel)

def getContours(img):
    """輪郭抽出

    Args:
        img: 元画像
    Returns:
        Array: 輪郭（0番目が外側、1番目が内側の円）
    """
    # 輪郭の検出
    contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if (len(contours) != 2):
        return False

    cont_0 = contours[0][:, 0]
    cont_1 = contours[1][:, 0]
    return cont_0, cont_1

def getMaxMin(cont):
    """輪郭の点群から最大・最小の座標を取得

    Args:
        cont: 輪郭の点群

    Returns:
        array: 最大・最小の座標（二次元配列）
    """
    cont_max_y = np.max(cont[:, 0])
    cont_max_x = np.max(cont[:, 1])
    cont_min_y = np.min(cont[:, 0])
    cont_min_x = np.min(cont[:, 1])
    return (cont_min_y, cont_min_x), (cont_max_y, cont_max_x)

def getCenter(cont_maxmin):
    """なげわの中心の取得

    Args:
        cont_maxmin: 輪郭の最大最小座標

    Returns:
        array: 中心のx, y
    """    
    center = np.average(cont_maxmin, 0)
    print(center)
    return center

def getRadius(cont, center):
    """内接円の半径を取得

    Args:
        cont: 輪郭の座標群
        center: 内接円中心の座標

    Returns:
        float: 内接円の半径
    """
    distance_arr = np.empty(1)
    for i in range(len(cont)):
        distance = np.linalg.norm(cont[i]-center)        
        distance_arr = np.append(distance_arr, distance)

    radius = np.min(distance_arr)
    print(radius)
    return radius
 

# 画像の読み込み
img = cv2.imread("./data/sample2.jpg", 0)
# 画像の二値化
img_opening = img2bin(img)

contours = getContours(img_opening)
if (contours == False) :
    exit

(cont_0, cont_1) = contours
cont_0_maxmin = getMaxMin(cont_0)
cont_1_maxmin = getMaxMin(cont_1)
center = getCenter(cont_1_maxmin)
radius = getRadius(cont_1, center)

img_dst = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
cv2.rectangle(img_dst, cont_0_maxmin[0], cont_0_maxmin[1], (255, 0, 0), 1)
cv2.rectangle(img_dst, cont_1_maxmin[0], cont_1_maxmin[1], (0, 0, 255), 1)
cv2.circle(img_dst, (int(center[0]), int(center[1])), 3, (255, 0, 255), 1)
cv2.circle(img_dst, (int(center[0]), int(center[1])), int(radius), (0, 255, 255), 1)

# 二値化画像の表示
cv2.imshow("img_src", img)
cv2.imshow("img_th_with_opening", img_opening)
cv2.imshow("img_dst", img_dst)
cv2.waitKey()
cv2.destroyAllWindows()
