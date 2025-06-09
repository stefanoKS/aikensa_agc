from calendar import c
import re
import stat
from unittest import result
from networkx import draw
import numpy as np
import cv2
import math
import yaml
import os
import pygame
import os
from PIL import ImageFont, ImageDraw, Image

from aikensa.scripts.scripts_img_processing import create_masks, draw_bounding_box, get_center, find_edge_point_mask, calclength, check_tolerance, check_id, draw_pitch_line
from aikensa.scripts.scripts_img_processing import draw_status_text_PIL

color = (0, 255, 0)
linecolor = (20,120,120)
text_offset = 40
endoffset_y = 0
bbox_offset = 1

pixelMultiplier = 0.161 #0.1592


border_width = 256


def partcheck(image, segmentationList):

    sorted_detections = sorted(sahi_predictionList, key=lambda d: d.bbox.minx)

    detectedid = []

    measuredPitch = []
    resultPitch = []
    deltaPitch = []

    resultid = []

    detectedposX = []
    detectedposY = []

    detectedWidth = []

    prev_center = None

    flag_pitch_furyou = 0
    flag_clip_furyou = 0
    flag_clip_hanire = 0
    flag_hole_notfound = 0
    leftmostPitch = 0
    rightmostPitch = 0

    status = "OK"
    print_status = ""

    combined_lmask = None
    ngreason = ""

    combined_lmask = None
    for lm in leftSegmentation:
        if lm.masks is not None:
            orig_shape = (image.shape[0] + border_width * 2 , segmentation_width + border_width * 2 )
            segmentation_xyn = lm.masks.xyn
            lmask = create_masks(segmentation_xyn, orig_shape)
            if combined_lmask is None:
                combined_lmask = np.zeros_like(lmask)
            combined_lmask = cv2.bitwise_or(combined_lmask, lmask)
            #resize back to original size
            combined_lmask = combined_lmask[border_width:-border_width, border_width:-border_width]
            combined_lmask = cv2.resize(combined_lmask, (segmentation_width, image.shape[0]))
            # cv2.imwrite("leftmask.jpg", combined_lmask)
        if lm.masks is None:
            status = "NG"
            print_status = "製品は見つかりません"
            image = draw_status_text_PIL(image, status, print_status, size="normal")

            resultPitch = [0] * (len(pitchSpec))
            measuredPitch = [0] * (len(pitchSpec))
            ngreason = "PART IS NOT FOUND"

            return image, measuredPitch, resultPitch, deltaPitch, status, ngreason
        
    combined_rmask = None
    for rm in rightSegmentation:
        if rm.masks is not None:
            orig_shape = (image.shape[0] + border_width * 2 , segmentation_width + border_width * 2 )
            segmentation_xyn = rm.masks.xyn
            rmask = create_masks(segmentation_xyn, orig_shape)
            if combined_rmask is None:
                combined_rmask = np.zeros_like(rmask)
            combined_rmask = cv2.bitwise_or(combined_rmask, rmask)
            #remove the pad from the image (pad size is 200 around the image)
            combined_rmask = combined_rmask[border_width:-border_width, border_width:-border_width]
            combined_rmask = cv2.resize(combined_rmask, (segmentation_width, image.shape[0]))
            # cv2.imwrite("rightmask.jpg", combined_rmask)
        if rm.masks is None:
            status = "NG"
            print_status = "製品は見つかりません"
            image = draw_status_text_PIL(image, status, print_status, size="small")

            resultPitch = [0] * (len(pitchSpec))
            measuredPitch = [0] * (len(pitchSpec))
            ngreason = "PART IS NOT FOUND"

            return image, measuredPitch, resultPitch, deltaPitch, status, ngreason

    combined_mask = np.zeros_like(image[:, :, 0])  # Single-channel black mask

    if combined_lmask is not None and combined_rmask is not None:
        combined_mask[:, segmentation_pixel_start:segmentation_pixel_finish] = combined_lmask
        combined_mask[:, -segmentation_pixel_finish:] = combined_rmask

    # cv2.imwrite("combined_mask.jpg", combined_mask)

    for i, detection in enumerate(sorted_detections):

        detectedid.append(detection.category.id)
        # print("Detected ID: ", detection.category.id)
        bbox = detection.bbox
        x, y = get_center(bbox)
        w = bbox.maxx - bbox.minx
        h = bbox.maxy - bbox.miny

        detectedposX.append(x)
        detectedposY.append(y)
        detectedWidth.append(w)


        center = draw_bounding_box(image, x, y, w, h, [image.shape[1], image.shape[0]], color=color)
      
        print (center)

        if prev_center is not None:
            length = calclength(prev_center, center)*pixelMultiplier
            measuredPitch.append(length)
        prev_center = center

    # print("Detected IDs: ", detectedid)


    if len(detectedposX) > 0:
        leftmostCenter = (detectedposX[0], detectedposY[0])
        leftmostWidth = detectedWidth[0]
        rightmostCenter = (detectedposX[-1], detectedposY[-1])
        rightmostWidth = detectedWidth[-1]
      
        # Positive Yoffsetval means going down, negative means going up
        left_edge = find_edge_point_mask(image, combined_mask, leftmostCenter, direction="left", Yoffsetval = -80, Xoffsetval = 0)
        right_edge = find_edge_point_mask(image, combined_mask, rightmostCenter, direction="right", Yoffsetval = -80, Xoffsetval = 0)

        leftmostPitch = calclength(leftmostCenter, left_edge)*pixelMultiplier
        rightmostPitch = calclength(rightmostCenter, right_edge)*pixelMultiplier

        #append the leftmost and rightmost pitch to the measuredPitch
        measuredPitch.insert(0, leftmostPitch)
        measuredPitch.append(rightmostPitch)
        #Reappend the leftmostcetner and rightmostcenter to the detectedposX and detectedposY
        detectedposX.insert(0, left_edge[0])
        detectedposY.insert(0, left_edge[1])
        detectedposX.append(right_edge[0])
        detectedposY.append(right_edge[1])


    #add total length
    #round the value to 1 decimal
    totalLength = sum(measuredPitch)
    measuredPitch.append(round(totalLength, 1))
    measuredPitch = [round(pitch, 1) for pitch in measuredPitch]

    if len(measuredPitch) == len(pitchSpec):
        resultPitch = check_tolerance(measuredPitch, pitchSpec, tolerance_pitch)
        resultid = check_id(detectedid, idSpec)

    if len(measuredPitch) != len(pitchSpec):
        resultPitch = [0] * len(pitchSpec)
        resultid = [0] * len(idSpec)
        measuredPitch = [0] * (len(pitchSpec))
        ngreason = "NUMBER OF CLIP MISMATCH"
        status = "NG"
        print_status = "クリップ数不足"

        image = draw_status_text_PIL(image, status, print_status, size="normal")

        return image, measuredPitch, resultPitch, resultid, status, ngreason


    if any(result != 1 for result in resultPitch):
        flag_pitch_furyou = 1
        status = "NG"
        ngreason = "CLIP PITCH NG"
        print_status = "クリップピッチ不良"

    # print("Resultpitch: ", resultPitch)
    # print("Resultid: ", resultid)
    # print("MeasuredPitch: ", measuredPitch)

    # if any(result != 1 for result in resultid):
    #     flag_clip_furyou = 1
    #     status = "NG"

    xy_pairs = list(zip(detectedposX, detectedposY))
    draw_pitch_line(image, xy_pairs, resultPitch, thickness=8)

    image = draw_status_text_PIL(image, status, print_status, size="normal")
    
    return image, measuredPitch, resultPitch, resultid, status, ngreason