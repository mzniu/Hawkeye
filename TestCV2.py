import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt

print(cv2.__version__)
print(sys.version)
MIN_MATCH_COUNT = 6


# img1 = cv2.imread('anchors/tattoo_seed.jpg',0)
# img2 = cv2.imread('anchors/hush.jpg',0)
# img1 = cv2.imread('./target.jpg',0)
# img2 = cv2.imread('./screenshot.jpg',0)
def get_match(image_path1, image2):
    img1 = cv2.imread(image_path1, 0)
    img2 = image2
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=16)
    search_params = dict(checks=500)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        a_x = np.int32(dst)[0][0][0]
        a_y = np.int32(dst)[0][0][1]
        b_x = np.int32(dst)[1][0][0]
        b_y = np.int32(dst)[1][0][1]
        c_x = np.int32(dst)[2][0][0]
        c_y = np.int32(dst)[2][0][1]
        d_x = np.int32(dst)[3][0][0]
        d_y = np.int32(dst)[3][0][1]
        x1 = max(a_x, b_x)
        y1 = max(a_y, d_y)
        x2 = min(c_x, d_x)
        y2 = min(b_y, c_y)
        img2 = cv2.polylines(img2, [np.int32([[[x1, y1]], [[x1, y2]], [[x2, y2]], [[x2, y1]]])], True, 255, 3,
                             cv2.LINE_AA)
        # return img2
    else:
        print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
        matchesMask = None
        # return img2
    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
    return img3
    # plt.imshow(img3, 'gray'),plt.show()
    # cv2.imshow("canny", img3)
    # cv2.waitKey()
    # cv2.destroyAllWindows()


# get_match('./target.jpg','./screenshot.jpg')



clicked = False


def onMouse(event, x, y, flags, param):
    global clicked
    if event == cv2.EVENT_LBUTTONUP:
        clicked = True


cameraCapture = cv2.VideoCapture(0)
cameraCapture.set(3, 2560)
cv2.namedWindow('MyWindow')
cv2.setMouseCallback('MyWindow', onMouse)

print('Showing camera feed. Click window or press any key to stop.')
success, frame = cameraCapture.read()
while cv2.waitKey(1) == -1 and not clicked:
    if frame is not None:
        cv2.imshow('MyWindow', get_match('./target.jpg', frame))  # cv2.imshow('MyWindow', frame)#
    success, frame = cameraCapture.read()
cameraCapture.release()
cv2.destroyWindow('MyWindow')
