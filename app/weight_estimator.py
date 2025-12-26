PIXEL_TO_GRAM_SCALE = 0.12  # grams per pixel^2 (calibrated)


def estimate_weight_grams(bbox):
    x1, y1, x2, y2 = bbox

    width = max(0, x2 - x1)
    height = max(0, y2 - y1)

    area = width * height
    weight_grams = area * PIXEL_TO_GRAM_SCALE

    return weight_grams
