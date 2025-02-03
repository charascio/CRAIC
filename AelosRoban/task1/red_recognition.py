import cv2
import numpy as np

def detect_red_objects(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 43, 46])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([156, 43, 46])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    red_objects = cv2.bitwise_and(image, image, mask=mask)

    _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #（x, y, w, h）
    red_objects_info = []

    for contour in contours:
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            red_objects_info.append(x)
            red_objects_info.append(y)
            red_objects_info.append(w)
            red_objects_info.append(h)

    return red_objects_info

def draw_rect(image, location):
    cv2.rectangle(image, (location[0], location[1]), (location[0] + location[2], location[1] + location[3]), (0, 255, 0), 2)
    cv2.circle(image, (int(location[0]+location[2]/2), int(location[1]+location[3]/2)), 5, (0, 0, 255), -1)  # 红色圆点表示中心

if __name__ == '__main__':
    image_path = 'red_ball.png'
    image = cv2.imread(image_path)
    red_objects_info = detect_red_objects(image)
    draw_rect(image, red_objects_info)
    cv2.imshow("img", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
