import cv2
import numpy as np


def cluster_density(cluster_mask):
    contours, _ = cv2.findContours(cluster_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(contour) for contour in contours]
    return sum(areas)


def combine_vertical_contours(contours):
    combined_contours = []
    skip = set()

    for i, cnt1 in enumerate(contours):
        if i in skip:
            continue

        x1, y1, w1, h1 = cv2.boundingRect(cnt1)
        combined_rect = [x1, y1, x1 + w1, y1 + h1]  # [left, top, right, bottom]

        for j, cnt2 in enumerate(contours):
            if i == j or j in skip:
                continue

            x2, y2, w2, h2 = cv2.boundingRect(cnt2)
            if (x1 <= x2 <= x1 + w1) or (x2 <= x1 <= x2 + w2):  # Check vertical overlap
                combined_rect[0] = min(combined_rect[0], x2)
                combined_rect[1] = min(combined_rect[1], y2)
                combined_rect[2] = max(combined_rect[2], x2 + w2)
                combined_rect[3] = max(combined_rect[3], y2 + h2)
                skip.add(j)

        combined_contours.append(combined_rect)

    return combined_contours


def preprocess_image(path):
    image = cv2.imread(path)
    sharp_filter = np.array([[-1, -1, -1],
                             [-1, 9, -1],
                             [-1, -1, -1]])
    image = cv2.filter2D(image, -1, kernel=sharp_filter)
    pixels = image.reshape(-1, 3)
    pixels = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 5
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    densities = []
    for i in range(k):
        cluster_mask = (labels == i).reshape(image.shape[:2]).astype(np.uint8) * 255
        density = cluster_density(cluster_mask)
        densities.append(density)
    selected_cluster = np.argmin(densities)

    mask = (labels == selected_cluster).reshape(image.shape[:2])

    masked_image = cv2.bitwise_and(image, image, mask=mask.astype(np.uint8) * 255)

    binary_image = \
        cv2.threshold(cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[
            1]
    binary_image = cv2.bitwise_not(binary_image)
    binary_image = cv2.erode(binary_image, np.ones((2, 2), np.uint8))
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
    combined_contours = combine_vertical_contours(contours)
    captcha_h = max(map(lambda c: c[3] - c[1], combined_contours))
    symbols = []
    for x1, y1, x2, y2 in combined_contours:
        symbol = binary_image[y1:y2, x1:x2]
        symbol = cv2.dilate(symbol, np.ones((2, 2), np.uint8))
        h = int(30 / captcha_h * (y2 - y1))
        w = int(h * ((x2 - x1) / (y2 - y1)))
        w = min(37, w)
        symbol = cv2.resize(symbol, (w, h), interpolation=cv2.INTER_AREA)
        reshaped = np.zeros((60, 40), dtype=np.uint8)
        reshaped[30 - h + 10:40, 3:3 + w] = symbol
        symbols.append(reshaped)
    return symbols


if __name__ == "__main__":
    preprocess_image("../generated/ZpiDWR.png")
