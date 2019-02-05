#the code is inspired from theses following sources :
# https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/ and
# https://github.com/jrosebr1/imutils/blob/master/imutils/object_detection.py
# https://github.com/techfort/pycv/blob/master/chapter7/car_detector/non_maximum.py

import numpy as np

def nms_average(boxes, scoreThresh=0.1):
    if len(boxes) == 0:
        return []

    # load all coordinate from all boxes, note: they are all sorted in advance when NMS_max is applied
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    score_arr = boxes[:, 4]

    # compute the area of the bounding boxes and sort the bounding boxes by the final scores
    deleted_idxs = np.where(score_arr[:len(score_arr)] < (score_arr[0]*(1 - scoreThresh)))[0]
    print(len(deleted_idxs))
    # average all the remaining windows and including scores.
    x1_avr = np.average((np.delete(x1, deleted_idxs)))
    y1_avr = np.average((np.delete(y1, deleted_idxs)))
    x2_avr = np.average((np.delete(x2, deleted_idxs)))
    y2_avr = np.average((np.delete(y2, deleted_idxs)))
    score_avr = np.average((np.delete(score_arr, deleted_idxs)))

    result_boxes = []
    result_boxes.append([x1_avr, y1_avr, x2_avr, y2_avr, score_avr])

    return result_boxes

#NMS-AVR for the case of many faces in a pictures, not working properly.
'''def nms_average_many(boxes, scoreThresh = 0.1, overlapThresh = 0.1):
    if len(boxes) == 0:
        return []

    # load all coordinate from all boxes, note: they are all sorted in advance when NMS_max is applied
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    score_arr = boxes[:, 4]
    # compute the area of the bounding boxes and sort the bounding boxes by the score, the box with highest score is at last
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = list(reversed(range(len(score_arr))))
    result_boxes = []
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last] #index of box has the highest score

        deleted_idxs = np.where(score_arr[:last] < (score_arr[i] * (1 - scoreThresh)))[0]
        idxs = np.delete(idxs, deleted_idxs)

        if len(idxs) == 0:
            break

        last = len(idxs) - 1
        i = idxs[last]

        pick = []

        # find the largest (x, y) coordinates for the start of the bounding box
        # and the smallest (x, y) coordinates for the end of the bounding box
        # in orther words, find all the coordinates of each intersection area between the picking box (the one with current highest score) and the rests.
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of all the intersection in order to compute their area
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # area of the last box that currently has the highest score.
        area_arr = np.zeros(len(idxs) - 1)
        area_arr.fill(area[i])

        # compute the  IOU of the last boxes and the rest
        overlap = (w * h) / (area[idxs[:last]] - w * h + area_arr)

        pick_idx =  np.where(overlap > overlapThresh)[0]
        # average all the remaining windows and including scores.
        x1_avr = np.average(x1[pick_idx])
        y1_avr = np.average(y1[pick_idx])
        x2_avr = np.average(x2[pick_idx])
        y2_avr = np.average(y2[pick_idx])
        score_avr = np.average(score_arr[pick_idx])

        result_boxes.append([x1_avr, y1_avr, x2_avr, y2_avr, score_avr])
        idxs = np.delete(idxs, np.concatenate(([last], pick_idx)))
    return result_boxes'''

#another way to compute NMS-AVR, not a good solution
'''
def nms_average_crappy(boxes, overlapThresh=0.2):
    result_boxes = []
    if len(boxes) == 0:
        return []
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the scores
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(boxes[:, 4])

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # area of i.
        area_i = np.maximum(0, x2[i] - x1[i] + 1) * np.maximum(0, y2[i] - y1[i] + 1)
        area_array = np.zeros(len(idxs) - 1)
        area_array.fill(area_i)
        # compute the ratio of overlap
        # overlap = (w * h) / (area[idxs[:last]]  - w * h + area_array)

        overlap = (w * h) / (area[idxs[:last]])
        delete_idxs = np.concatenate(([last], np.where(overlap > overlapThresh)[0]))
        xmin = 10000
        ymin = 10000
        xmax = 0
        ymax = 0
        ave_prob = 0
        width = x2[i] - x1[i] + 1
        height = y2[i] - y1[i] + 1
        for idx in delete_idxs:
            ave_prob += boxes[idxs[idx]][4]
            if boxes[idxs[idx]][0] < xmin:
                xmin = boxes[idxs[idx]][0]
            if boxes[idxs[idx]][1] < ymin:
                ymin = boxes[idxs[idx]][1]
            if boxes[idxs[idx]][2] > xmax:
                xmax = boxes[idxs[idx]][2]
            if boxes[idxs[idx]][3] > ymax:
                ymax = boxes[idxs[idx]][3]
        if x1[i] - xmin > 0.1 * width:
            xmin = x1[i] - 0.1 * width
        if y1[i] - ymin > 0.1 * height:
            ymin = y1[i] - 0.1 * height
        if xmax - x2[i] > 0.1 * width:
            xmax = x2[i] + 0.1 * width
        if ymax - y2[i] > 0.1 * height:
            ymax = y2[i] + 0.1 * height
        result_boxes.append([xmin, ymin, xmax, ymax, ave_prob / len(delete_idxs)])
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, delete_idxs)

    # return only the bounding boxes that were picked using the
    # integer data type
    # result = np.delete(boxes[pick],np.where(boxes[pick][:, 4] < 0.9)[0],  axis=0)
    # print boxes[pick]
    return result_boxes
'''

def nms_max(boxes, overlapThresh=0.3):
    if len(boxes) == 0:
        return []
    # initialize the list of picked indexes
    pick = []
    # load all coordinate from all boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # compute the area of the bounding boxes and sort the bounding boxes by the score, the box with highest score is at last
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(boxes[:, 4])

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the bounding box
        # and the smallest (x, y) coordinates for the end of the bounding box
        # in orther words, find all the coordinates of each intersection area between the picking box (the one with current highest score) and the rests.
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of all the intersection in order to compute their area
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # area of the last box that currently has the highest score.
        area_arr = np.zeros(len(idxs) - 1)
        area_arr.fill(area[i])

        # compute the  IOU of the last boxes and the rest
        overlap = (w * h) / (area[idxs[:last]] - w * h + area_arr)

        # delete boxes based on set Threshold
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return boxes[pick]