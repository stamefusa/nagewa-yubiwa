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

# 画像の読み込み
img = cv2.imread("./data/sample.jpg", 0)
img_opening = img2bin(img)

# 二値化画像の表示
cv2.imshow("img_th_with_opening", img_opening)
cv2.imshow("img_org", img)
cv2.waitKey()
cv2.destroyAllWindows()
