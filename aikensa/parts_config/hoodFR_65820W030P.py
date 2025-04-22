import stat
import numpy as np
import cv2
import math
import yaml
import os
import pygame
import os
from PIL import ImageFont, ImageDraw, Image

pygame.mixer.init()
ok_sound = pygame.mixer.Sound("aikensa/sound/positive_interface.wav") 
ok_sound_v2 = pygame.mixer.Sound("aikensa/sound/mixkit-software-interface-remove-2576.wav")
ng_sound = pygame.mixer.Sound("aikensa/sound/mixkit-classic-short-alarm-993.wav")  
ng_sound_v2 = pygame.mixer.Sound("aikensa/sound/mixkit-system-beep-buzzer-fail-2964.wav")
kanjiFontPath = "aikensa/font/NotoSansJP-ExtraBold.ttf"

pitchSpec = [26, 107, 75, 75, 75, 75, 75, 92, 75, 102, 100, 129, 103, 109, 103, 129, 100, 102, 75, 92, 75, 75, 75, 75, 75, 107, 26]
idSpec = [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
tolerance_pitch = [3.0] * 27
tolerance_pitch[0] = 3.5
tolerance_pitch[-1] = 3.5

color = (0, 255, 0)
text_offset = 40
endoffset_y = 0
bbox_offset = 10

pixelMultiplier = 0.1590


def partcheck(image, sahi_predictionList, leftSegmentation, rightSegmentation):

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
    ngreason = ""

    # cannydetection_image = image.copy() #Make sure to copy the image to avoid modifying the original image

    combined_lmask = None
    for lm in leftSegmentation:
        if lm.masks is not None:
            orig_shape = (image.shape[0], 1024)
            segmentation_xyn = lm.masks.xyn
            lmask = create_masks(segmentation_xyn, orig_shape)
            if combined_lmask is None:
                combined_lmask = np.zeros_like(lmask)
            combined_lmask = cv2.bitwise_or(combined_lmask, lmask)
            # cv2.imwrite("leftmask.jpg", combined_lmask)

        if lm.masks is None:
            print_status = print_status + " 製品認識不良"
            status = "NG"
            resultPitch = [0] * (len(pitchSpec))
            measuredPitch = [0] * (len(pitchSpec))
            resultid = [0] * (len(idSpec))
            image = draw_status_text_PIL(image, status, print_status, size="normal")
            ngreason = "PART NOT FOUND"

            return image, measuredPitch, resultPitch, resultid, status, ngreason

    combined_rmask = None
    for rm in rightSegmentation:
        if rm.masks is not None:
            orig_shape = (image.shape[0], 1024)
            segmentation_xyn = rm.masks.xyn
            rmask = create_masks(segmentation_xyn, orig_shape)
            if combined_rmask is None:
                combined_rmask = np.zeros_like(rmask)
            combined_rmask = cv2.bitwise_or(combined_rmask, rmask)
            # cv2.imwrite("rightmask.jpg", combined_rmask)

        if lm.masks is None:
            print_status = print_status + " 製品認識不良"
            status = "NG"
            resultPitch = [0] * (len(pitchSpec))
            measuredPitch = [0] * (len(pitchSpec))
            resultid = [0] * (len(idSpec))
            ngreason = "PART NOT FOUND"
            image = draw_status_text_PIL(image, status, print_status, size="normal")

            return image, measuredPitch, resultPitch, resultid, status, ngreason

    combined_mask = np.zeros_like(image[:, :, 0])  # Single-channel black mask

    if combined_lmask is not None and combined_rmask is not None:
        combined_mask[:, :1024] = combined_lmask 
        combined_mask[:, -1024:] = combined_rmask 
        # cv2.imwrite("combined_mask.jpg", combined_mask)

    for i, detection in enumerate(sorted_detections):
        detectedid.append(detection.category.id)
        if detection.category.id == 0:
            bbox = detection.bbox
            x, y = get_center(bbox)
            w = bbox.maxx - bbox.minx
            h = bbox.maxy - bbox.miny
            # class_name = detection.category.name

            detectedposX.append(x)
            detectedposY.append(y)
            detectedWidth.append(w)

            #id 0 object is white clip
            #id 1 object is holes (not implemented yet)
            center = draw_bounding_box(image, x, y, w, h, [image.shape[1], image.shape[0]], color=color)

            if prev_center is not None:
                length = calclength(prev_center, center)*pixelMultiplier
                measuredPitch.append(length)
            prev_center = center

    #Check if detectedposX is not empty
    if len(detectedposX) > 0:
        leftmostCenter = (detectedposX[0], detectedposY[0])
        leftmostWidth = detectedWidth[0]
        rightmostCenter = (detectedposX[-1], detectedposY[-1])
        rightmostWidth = detectedWidth[-1]
        adjustment_offset = 5 # to make sure it goes above the clip itself
        # left_edge = find_edge_point(cannydetection_image, leftmostCenter, direction="left", Yoffsetval = 0, Xoffsetval = leftmostWidth + adjustment_offset)
        # right_edge = find_edge_point(cannydetection_image, rightmostCenter, direction="right", Yoffsetval = 0, Xoffsetval = rightmostWidth + adjustment_offset)

        left_edge = find_edge_point_mask(image, combined_mask, leftmostCenter, direction="left", Yoffsetval = 0, Xoffsetval = 0)
        right_edge = find_edge_point_mask(image, combined_mask, rightmostCenter, direction="right", Yoffsetval = 0, Xoffsetval = 0)

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

    #round the value to 1 decimal
    measuredPitch = [round(pitch, 1) for pitch in measuredPitch]

    if len(measuredPitch) == len(pitchSpec):

        resultPitch = check_tolerance(measuredPitch, pitchSpec, tolerance_pitch)
        resultid = check_id(detectedid, idSpec)

    if len(measuredPitch) != len(pitchSpec):

        resultPitch = [0] * (len(pitchSpec))
        measuredPitch = [0] * (len(pitchSpec))
        resultid = [0] * (len(idSpec))
        status = "NG"
        ngreason = "NUMBER OF CLIP MISMATCH"

        return image, measuredPitch, resultPitch, resultid, status, ngreason


    if any(result != 1 for result in resultPitch):
        flag_pitch_furyou = 1
        status = "NG"
        ngreason = "CLIP PITCH NG"

    # if any(result != 1 for result in resultid):
    #     flag_clip_furyou = 1
    #     status = "NG"

    xy_pairs = list(zip(detectedposX, detectedposY))
    draw_pitch_line(image, xy_pairs, resultPitch, thickness=8)
    
    return image, measuredPitch, resultPitch, resultid, status, ngreason

def create_masks(segmentation_result, orig_shape):
    mask = np.zeros((orig_shape[0], orig_shape[1]), dtype=np.uint8)
    for polygon in segmentation_result:
        polygon = np.array([[int(x * orig_shape[1]), int(y * orig_shape[0])] for x, y in polygon], dtype=np.int32)
        cv2.fillPoly(mask, [polygon], 255)
    return mask

def play_sound(status):
    if status == "OK":
        # ok_sound.play()
        ok_sound_v2.play()
    elif status == "NG":
        # ng_sound.play()
        ng_sound_v2.play()

def get_center(bbox):
    center_x = bbox.minx + (bbox.maxx - bbox.minx) / 2
    center_y = bbox.miny + (bbox.maxy - bbox.miny) / 2
    return center_x, center_y

def print_bbox_structure(bbox):
    print(f"BoundingBox attributes: {dir(bbox)}")

def draw_flag_status(image, flag_pitchfuryou, flag_clip_furyou, flag_clip_hanire):
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(kanjiFontPath, 40)
    color=(200,10,10)
    if flag_pitchfuryou == 1:
        draw.text((120, 10), u"クリップピッチ不良", font=font, fill=color)  
    if flag_clip_furyou == 1:
        draw.text((120, 60), u"クリップ類不良", font=font, fill=color)  
    if flag_clip_hanire == 1:
        draw.text((120, 110), u"クリップ半入れ", font=font, fill=color)
    
    # Convert back to BGR for OpenCV compatibility
    image = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    return image

def check_id(detectedid, idSpec):
    result = [0] * len(idSpec)
    for i, (spec, detected) in enumerate(zip(idSpec, detectedid)):
        if spec == detected:
            result[i] = 1
    return result

def draw_pitch_line(image, xy_pairs, pitchresult, thickness=2):
    xy_pairs = [(int(x), int(y)) for x, y in xy_pairs]

    if len(xy_pairs) != 0:
        for i in range(len(xy_pairs) - 1):
            if i < len(pitchresult) and pitchresult[i] is not None:
                if pitchresult[i] == 1:
                    lineColor = (0, 255, 0)
                else:
                    lineColor = (255, 0, 0)

                cv2.line(image, xy_pairs[i], xy_pairs[i+1], lineColor, thickness)
                

    return None


#add "OK" and "NG"
def draw_status_text(image, status, size = "normal"):
    # Define the position for the text: Center top of the image
    center_x = image.shape[1] // 2
    if size == "normal":
        top_y = 50  # Adjust this value to change the vertical position
        font_scale = 5.0  # Increased font scale for bigger text

    elif size == "small":
        top_y = 10
        font_scale = 2.0  # Increased font scale for bigger text
    

    # Text properties
    
    font_thickness = 8  # Increased font thickness for bolder text
    outline_thickness = font_thickness + 2  # Slightly thicker for the outline
    text_color = (255, 0, 0) if status == "NG" else (0, 255, 0)  # Red for NG, Green for OK
    outline_color = (0, 0, 0)  # Black for the outline

    # Calculate text size and position
    text_size, _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    text_x = center_x - text_size[0] // 2
    text_y = top_y + text_size[1]

    # Draw the outline
    cv2.putText(image, status, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, outline_color, outline_thickness)

    # Draw the text over the outline
    cv2.putText(image, status, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)

    return image


def check_tolerance(checkedPitchResult, pitchSpec, pitchTolerance):
    result = [0] * len(pitchSpec)
    for i, (spec, detected) in enumerate(zip(pitchSpec, checkedPitchResult)):
        if abs(spec - detected) <= pitchTolerance[i]:
            result[i] = 1
    return result

def yolo_to_pixel(yolo_coords, img_shape):
    class_id, x, y, w, h, confidence = yolo_coords
    x_pixel = int(x * img_shape[1])
    y_pixel = int(y * img_shape[0])
    return x_pixel, y_pixel

def find_edge_point_mask(image, mask, center, direction="None", Xoffsetval = 0, Yoffsetval = 0):
    x, y = center[0], center[1]

    min_x = 0
    max_x = image.shape[1] - 1

    if direction == "left":
        while x - Xoffsetval >= 0:
            if mask[int(y + Yoffsetval), int(x - Xoffsetval)] == 0:  # Found an edge
                return x - Xoffsetval, y
            x -= 1
        return min_x, y

    if direction == "right":
        while x + Xoffsetval < image.shape[1]:
            if mask[int(y + Yoffsetval), int(x + Xoffsetval)] == 0:  # Found an edge
                return x + Xoffsetval, y
            x += 1
        return max_x, y

    return None  # If an invalid direction is provided

def find_edge_point(image, center, direction="None", Xoffsetval = 0, Yoffsetval = 0):
    x, y = center[0], center[1]
    blur = 11
    brightness = 0
    contrast = 3.0
    lower_canny = 15
    upper_canny = 110

    # Apply adjustments
    adjusted_image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
    gray_image = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (blur | 1, blur | 1), 0)
    canny_img = cv2.Canny(blurred_image, lower_canny, upper_canny)

    # cv2.imwrite(f"1adjusted_image_{direction}.jpg", adjusted_image)
    # cv2.imwrite(f"2gray_image_{direction}.jpg", gray_image)
    # cv2.imwrite(f"3blurred_image_{direction}.jpg", blurred_image)
    # cv2.imwrite(f"4canny_debug_{direction}.jpg", canny_img)
    min_x = 0
    max_x = image.shape[1] - 1

    if direction == "left":
        while x - Xoffsetval >= 0:
            if canny_img[int(y + Yoffsetval), int(x - Xoffsetval)] == 255:  # Found an edge
                return x - Xoffsetval, y
            x -= 1
        return min_x, y

    if direction == "right":
        while x + Xoffsetval < image.shape[1]:
            if canny_img[int(y + Yoffsetval), int(x + Xoffsetval)] == 255:  # Found an edge
                return x + Xoffsetval, y
            x += 1
        return max_x, y

    return None  # If an invalid direction is provided

def drawcircle(image, pos, class_id): #for ire and hanire
    #draw either green or red circle depends on the detection
    if class_id == 0:
        color = (60, 200, 60)
    elif class_id == 1:
        color = (60, 60, 200)
    #check if pos is tupple
    pos = (int(pos[0]), int(pos[1]))

    cv2.circle(img=image, center=pos, radius=30, color=color, thickness=2, lineType=cv2.LINE_8)

    return image

def drawbox(image, pos, length, offset = text_offset, font_scale=1.7, font_thickness=4):
    pos = (pos[0], pos[1])
    rectangle_bgr = (255, 255, 255)
    (text_width, text_height), _ = cv2.getTextSize(f"{length:.2f}", cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    
    top_left_x = pos[0] - text_width // 2 - 8
    top_left_y = pos[1] - text_height // 2 - 8 - offset
    bottom_right_x = pos[0] + text_width // 2 + 8
    bottom_right_y = pos[1] + text_height // 2 + 8 - offset
    
    cv2.rectangle(image, (top_left_x, top_left_y),
                  (bottom_right_x, bottom_right_y),
                  rectangle_bgr, -1)
    
    return image

def drawtext(image, pos, length, font_scale=1.7, offset = text_offset, font_thickness=6):
    pos = (pos[0], pos[1])
    font_scale = font_scale
    text = f"{length:.1f}"
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    
    text_x = pos[0] - text_width // 2
    text_y = pos[1] + text_height // 2 - offset
    
    cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (20, 125, 20), font_thickness)
    return image

def calclength(p1, p2):
    length = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    return length

def draw_bounding_box(image, x, y, w, h, img_size, color=(0, 255, 0), thickness=3, bbox_offset=bbox_offset):
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)

    x1, y1 = int(x - w // 2) - bbox_offset, int(y - h // 2) - bbox_offset
    x2, y2 = int(x + w // 2) + bbox_offset, int(y + h // 2) + bbox_offset
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    center_x, center_y = x, y
    return (center_x, center_y)

# class BoundingBox:
#     def __init__(self, minx, miny, maxx, maxy):
#         self.minx = minx
#         self.miny = miny
#         self.maxx = maxx
#         self.maxy = maxy

# class PredictionScore:
#     def __init__(self, value):
#         self.value = value

# class Category:
#     def __init__(self, id, name):
#         self.id = id
#         self.name = name

# class ObjectPrediction:
#     def __init__(self, bbox, score, category):
#         self.bbox = bbox
#         self.score = score
#         self.category = category