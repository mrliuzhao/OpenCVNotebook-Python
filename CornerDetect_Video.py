import cv2


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Harris角点检测
    dst = cv2.cornerHarris(gray, 3, 23, 0.04)
    harrisCorner = frame.copy()
    harrisCorner[dst > 0.01 * dst.max()] = [0, 0, 255]

    # SIFT特征检测，收费
    # sift = cv2.xfeatures2d.SIFT_create()
    # keypoints, descriptors = sift.detectAndCompute(gray, mask=None)
    # siftCorner = frame.copy()
    # cv2.drawKeypoints(siftCorner, keypoints, siftCorner, (0, 0, 255),
    #                   cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # SURF特征检测，收费
    # surf = cv2.xfeatures2d.SURF_create()
    # keypoints, descriptors = surf.detectAndCompute(gray, mask=None)
    # surfCorner = frame.copy()
    # cv2.drawKeypoints(surfCorner, keypoints, surfCorner, (0, 0, 255),
    #                   cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # ORB模型
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(gray, mask=None)
    orbCorner = frame.copy()
    # cv2.drawKeypoints(orbCorner, keypoints, orbCorner, (0, 0, 255),
    #                   cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.drawKeypoints(orbCorner, keypoints, orbCorner, (0, 0, 255),
                      cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)


    cv2.imshow('Display', cv2.hconcat((harrisCorner, orbCorner)))
    if cv2.waitKey(1) == 27:
        break


cap.release()
cv2.destroyAllWindows()


