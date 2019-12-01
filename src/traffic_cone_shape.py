import cv2
import numpy as np


def isHullACone(each):
    each_hull_point = []

    x, y, w, h = cv2.boundingRect(each)
    # check if the aspect raitio
    aspect_ration = w / h
    print(aspect_ration)

    if aspect_ration > 0.8:
        pass

    y_center = y + int(h / 2.0)
    print(f"center:{y_center}")

    # list of points above the center and Below the centers
    # print(each.shape)
    # print(each[:,0,0], each[:,0])

    list_of_points_above_center = []
    list_of_points_below_center = []
    for x, y in each[:, 0]:
        print(x, y)
        if y < y_center:
            list_of_points_below_center.append([x, y])
        else:
            list_of_points_above_center.append([x, y])

    print('Upper', list_of_points_above_center)
    print('lower', list_of_points_below_center)

    left_most = list_of_points_below_center[0][0]
    right_most = list_of_points_below_center[0][0]

    print("First", left_most, right_most)
    for xx, yy in list_of_points_below_center:
        print("X, y", xx, yy)
        if x < left_most:
            left_most = xx
        print(xx > left_most, xx, left_most)
        if xx > left_most:
            print("Setting", right_most, "to", x)
            right_most = xx

    print("Last- left most and right most", left_most, right_most)
    flag = True
    for x_v, y_v in list_of_points_above_center:
        print("Above", x_v, y_v)
        if x_v < left_most or x_v > right_most:
            flag = False
            return False
        else:
            flag = True
    print("Finally Returning", flag)
    return flag


print(cv2.__version__)
img = cv2.imread("../Data/Images/1.jpg")

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

imgLowThreshold = cv2.inRange(hsv, np.array([0, 135, 135]), np.array([15, 255, 255]))
imgHighThreshold = cv2.inRange(hsv, np.array([159, 135, 135]), np.array([179, 255, 255]))

cv2.imshow("test", img)
# cv2.imshow("Thresh1", imgLowThreshold)
# cv2.imshow("Thresh2", imgHighThreshold)

imgThresh = cv2.bitwise_or(imgLowThreshold, imgHighThreshold)

# cv2.imshow("thresh", imgThresh)


img_smooth = imgThresh.copy()

kernel = np.ones((1, 1), np.uint8)

cv2.erode(img_smooth, kernel=kernel)
cv2.dilate(img_smooth, kernel=kernel, iterations=1)

cv2.GaussianBlur(img_smooth, (3, 3), 0)

# cv2.imshow("imThreshGaussBlur", img_copy)

canny = cv2.Canny(img_smooth, 160.0, 80.0)

cv2.imshow("canny", canny)

canny_copy = canny.copy()

# find contours on Canny data
contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# cv2.drawContours(canny, contours, 0, (0,255,0), 10)

convx_hull = []
imgContour = imgThresh.copy()

print(contours[0].reshape(-1, 2))

for i, c in enumerate(contours):

    # print(c.shape, c.reshape(-1, 2).shape[0], c.reshape(-1, 2).shape[1])
    approx = cv2.approxPolyDP(c, 0.01*cv2.arcLength(c, True), True)  # 0.01*cv2.arcLength(c, True)

    cv2.drawContours(canny, [approx], 0, (255, 255, 0), 5) #, lineType=cv2.LINE_AA, offset=(0,0)
    # cv2.drawContours(canny, c, 0, (255, 0, 0), 3)
    hull = cv2.convexHull(c, False, clockwise=True)
    #
    # print(hull.shape, hull)
    #
    # if i ==1:
    #     print(i,type(hull), hull.shape, hull.shape[0])
    #     cv2.drawContours(img, hull, 0, (220, 20, 60), 15)
    #     convx_hull.append(hull)
    # else:
    #     print(i,type(hull), hull.shape, hull.shape[0])
    #     cv2.drawContours(img, hull, 0, (20, 20, 60), 25)
    #     convx_hull.append(hull)
    #
    # cv2.drawContours(canny, [hull], 0, (i+30, 78, 23), 10)

    # cv2.putText(canny, i, (130, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    # print(hull.shape)
    if hull.shape[0] >= 3 and hull.shape[0] <= 10:
        # cv2.circle(canny, (c[0], c[1]), 1, (255, 0, 0), 3)
        # print("Loop:", hull.shape, hull.shape[0], len(hull))
        if isHullACone(hull):
            print("Yes a Hull")
            cv2.drawContours(canny, [hull], 0, (174, 90, 120), 12)
            convx_hull.append(hull)

    # break

# approx = cv2.approxPolyDP(contours, 0.01*cv2.arcLength(contours, True), True)
# print(approx)
# for i in range(len(contours)):
#     # print("one")
#       # 0.01*cv2.arcLength(c, True)
#     cv2.drawContours(imgContour, contours, i, (255, 0, 0), 3) #, lineType=cv2.LINE_AA, offset=(0,0)
#     # cv2.drawContours(imgContour, [c], 0, (255, 0, 0), 3)

cv2.imshow("canny after plotting contours", canny)


# cv2.imshow("erroded", erroded)
# cv2.imshow("dilation", dilation)
# cv2.imshow("gauss_blur", gauss_blur)
#
# cv2.imshow("canny_copy", canny_copy)



cv2.waitKey(0)
cv2.destroyAllWindows()
