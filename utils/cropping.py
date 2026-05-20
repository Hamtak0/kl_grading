center = "_FTG"
other = ["_FH", "_TPE_lat", "_DF_lat", "_TPE_med", "_DF_med"]

def center_to_corners(center_x: float, center_y: float, height: float, width: float) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]:
    x_min = center_x - width / 2
    x_max = center_x + width / 2
    # left top, left bottom, right bottom, right top
    return (x_min, center_y - height / 2), (x_min, center_y + height / 2), (x_max, center_y + height / 2), (x_max, center_y - height / 2)

def get_all_points(data: dict) -> dict[str, tuple[float, float]]:
    points = {}
    # ignore the third position
    for block in data['markups'][0]['controlPoints']:
        points[block['label']] = (block['position'][0], block['position'][1])
    return points

def cropping_points(data: dict, extra_margin: float = 0.25) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
    # positions were in markups/[0]/controlPoints/{label, position}
    # position does contain 3 dimension reduce the third one
    # label contains name of the point also add side "L", "R"

    points = get_all_points(data)

    # left side
    left_labels = ["L" + label for label in other]
    cx_l, cy_l = points["L" + center]
    
    # calculate maximum distance from the center in X and Y directions
    max_dx_l = max(abs(points[label][0] - cx_l) for label in left_labels)
    max_dy_l = max(abs(points[label][1] - cy_l) for label in left_labels)
    
    # base dimension is 2x the max distance, then multiply the extra margin percentage
    w_l = (2 * max_dx_l) * (1 + extra_margin)
    h_l = (2 * max_dy_l) * (1 + extra_margin)

    # right side
    right_labels = ["R" + label for label in other]
    cx_r, cy_r = points["R" + center]
    
    max_dx_r = max(abs(points[label][0] - cx_r) for label in right_labels)
    max_dy_r = max(abs(points[label][1] - cy_r) for label in right_labels)
    
    w_r = (2 * max_dx_r) * (1 + extra_margin)
    h_r = (2 * max_dy_r) * (1 + extra_margin)

    return (cx_l, cy_l, h_l, w_l), (cx_r, cy_r, h_r, w_r)
