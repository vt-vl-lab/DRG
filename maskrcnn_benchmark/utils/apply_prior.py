import numpy as np


def apply_prior(Object, prediction):
    if Object != 32:  # not a snowboard, then the action is impossible to be snowboard
        prediction[21] = 0

    if Object != 74:  # not a book, then the action is impossible to be read
        prediction[24] = 0

    if Object != 33:  # not a sports ball, then the action is impossible to be kick
        prediction[7] = 0

    if (Object != 41) and (Object != 40) and (Object != 42) and (
            Object != 46):  # not 'wine glass', 'bottle', 'cup', 'bowl', then the action is impossible to be drink
        prediction[13] = 0

    if Object != 37:  # not a skateboard, then the action is impossible to be skateboard
        prediction[26] = 0

    if Object != 38:  # not a surfboard, then the action is impossible to be surfboard
        prediction[0] = 0

    if Object != 31:  # not a ski, then the action is impossible to be ski
        prediction[1] = 0

    if Object != 64:  # not a laptop, then the action is impossible to be work on computer
        prediction[8] = 0

    if (Object != 77) and (Object != 43) and (
            Object != 44):  # not 'scissors', 'fork', 'knife', then the action is impossible to be cur instr
        prediction[2] = 0

    if (Object != 33) and (
            Object != 30):  # not 'sports ball', 'frisbee', then the action is impossible to be throw and catch
        prediction[15] = 0
        prediction[28] = 0

    if Object != 68:  # not a cellphone, then the action is impossible to be talk_on_phone
        prediction[6] = 0

    if (Object != 14) and (Object != 61) and (Object != 62) and (Object != 60) and (Object != 58) and (
            Object != 57):  # not 'bench', 'dining table', 'toilet', 'bed', 'couch', 'chair', then the action is impossible to be lay
        prediction[12] = 0

    if (Object != 32) and (Object != 31) and (Object != 37) and (
            Object != 38):  # not 'snowboard', 'skis', 'skateboard', 'surfboard', then the action is impossible to be jump
        prediction[11] = 0

    if (Object != 47) and (Object != 48) and (Object != 49) and (Object != 50) and (Object != 51) and (
            Object != 52) and (Object != 53) and (Object != 54) and (Object != 55) and (
            Object != 56):  # not ''banana', 'apple', 'sandwich', 'orange', 'carrot', 'broccoli', 'hot dog', 'pizza', 'cake', 'donut', then the action is impossible to be eat_obj
        prediction[9] = 0

    if (Object != 43) and (Object != 44) and (
            Object != 45):  # not 'fork', 'knife', 'spoon', then the action is impossible to be eat_instr
        prediction[16] = 0

    if (Object != 39) and (
            Object != 35):  # not 'tennis racket', 'baseball bat', then the action is impossible to be hit_instr
        prediction[19] = 0

    if (Object != 33):  # not 'sports ball, then the action is impossible to be hit_obj
        prediction[20] = 0

    if (Object != 2) and (Object != 4) and (Object != 6) and (Object != 8) and (Object != 9) and (Object != 7) and (
            Object != 5) and (Object != 3) and (Object != 18) and (
            Object != 21):  # not 'bicycle', 'motorcycle', 'bus', 'truck', 'boat', 'train', 'airplane', 'car', 'horse', 'elephant', then the action is impossible to be ride
        prediction[5] = 0

    if (Object != 2) and (Object != 4) and (Object != 18) and (Object != 21) and (Object != 14) and (Object != 57) and (
            Object != 58) and (Object != 60) and (Object != 62) and (Object != 61) and (Object != 29) and (
            Object != 27) and (
            Object != 25):  # not 'bicycle', 'motorcycle', 'horse', 'elephant', 'bench', 'chair', 'couch', 'bed', 'toilet', 'dining table', 'suitcase', 'handbag', 'backpack', then the action is impossible to be sit
        prediction[10] = 0

    if (Object == 1):
        prediction[4] = 0

    return prediction


def apply_prior_Graph(O_class, prediction_HO):
    prediction_HO_mask = np.empty((0, 29), dtype=np.float32)

    for idx in range(len(prediction_HO)):
        prediction = prediction_HO[idx]
        Object = O_class[idx][0]
        prediction = apply_prior(Object, prediction)
        prediction_HO_mask = np.concatenate((prediction_HO_mask, prediction.reshape(1, 29)), axis=0)

    return prediction_HO_mask
