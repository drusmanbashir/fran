
def to_even(input_num, lower=True):
    np.fnc = np.subtract if lower == True else np.add
    output_num = np.fnc(input_num, input_num % 2)
    return int(output_num)


def bbox_bg_only(bbox_stats):
    all_fg_bbox = [bb for bb in bbox_stats if bb["label"] == "all_fg"][0]
    bboxes = all_fg_bbox["bounding_boxes"]
    if len(bboxes) == 1:
        return True
    elif bboxes[0] != bboxes[1]:
        return False
    else:
        tr()


