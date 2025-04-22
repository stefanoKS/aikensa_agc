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


pitchSpecLH = [85, 87, 98, 98, 78, 113, 103]
pitchSpecRH = [103, 113, 78, 98, 98, 87, 85]
pitchToleranceLH = [2.0, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]
pitchToleranceRH = [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 2.0]
clipSpecLH = [2, 1, 0, 0, 0, 0, 3, 3, 0, 1] #white is 0, brown is 1, yellow is 2, orange is 3
clipSpecRH = [0, 1, 3, 3, 1, 1, 1, 1, 0, 2] #white is 0, brown is 1, yellow is 2, orange is 3

dailytenken01Spec = [100, 100, 100, 100, 100]
dailytenken01Tolerance = [1.5, 1.5, 1.5, 1.5, 1.5]
dailytenken01ClipSpec = [0, 1, 3, 0, 1, 3]

dailytenken02ClipSpec = [2, 0, 1, 3]

pitchSpecKatabu = [14]
pitchToleranceKatabu = [1.5]

pixelMultiplier = 0.19968499 #basically multiplier from 1/arucoplanarize param -> will create a constant for this later
pixelMultiplier_katabumarking = 0.2

text_offset = 40

endoffset_y = 0
bbox_offset = 10



def partcheck(img, img_katabumarking, detections, katabumarking_detection, hanire_detection, partid=None):

    sorted_detections = sorted(detections, key=lambda d: d.bbox.minx)


    middle_lengths = []
    katabumarking_lengths = []

    detectedid = []
    customid = []

    detectedPitch = []
    deltaPitch = []
    deltaPitchKatabu = []

    detectedposX = []
    detectedposY = []


    detectedposX_katabumarking = []
    detectedposY_katabumarking = []

    pitchresult = []
    checkedPitchResult = []

    katabupitchresult = []

    allpitchresult = []

    prev_center = None
    prev_center_katabumarking = None

    flag_pitchfuryou = 0
    flag_clip_furyou = 0
    flag_clip_hanire = 0

    #KATABU MARKING DETECTION
    #class 0 is for clip, class 1 is for katabu marking
    for r in katabumarking_detection:
        for box in r.boxes:
            x_marking, y_marking = float(box.xywh[0][0].cpu()), float(box.xywh[0][1].cpu())
            w_marking, h_marking = float(box.xywh[0][2].cpu()), float(box.xywh[0][3].cpu())
            class_id_marking = int(box.cls.cpu())

            # print(class_id_marking)

            if class_id_marking == 0:
                color = (0, 255, 0)
            elif class_id_marking == 1:
                color = (100, 100, 200)

            center_katabummarking = draw_bounding_box(img_katabumarking, 
                                       x_marking, y_marking, 
                                       w_marking, h_marking, 
                                       [img_katabumarking.shape[1], img_katabumarking.shape[0]], color=color,
                                       bbox_offset=3, thickness=2)
            
            if class_id_marking == 1:
                if partid == "LH":
                    center_katabummarking = (int(x_marking - w_marking/2), int(y_marking))
                elif partid == "RH":
                    center_katabummarking = (int(x_marking + w_marking/2), int(y_marking))
            
            if prev_center_katabumarking is not None:
                length = calclength(prev_center_katabumarking, center_katabummarking)*pixelMultiplier_katabumarking
                katabumarking_lengths.append(length)
                line_center = ((prev_center_katabumarking[0] + center_katabummarking[0]) // 2, (prev_center_katabumarking[1] + center_katabummarking[1]) // 2)
                img_katabumarking = drawbox(img_katabumarking, line_center, length, font_scale=0.8, offset=40, font_thickness=2)
                img_katabumarking = drawtext(img_katabumarking, line_center, length, font_scale=0.8, offset=40, font_thickness=2)

            prev_center_katabumarking = center_katabummarking

            detectedposX_katabumarking.append(center_katabummarking[0])
            detectedposY_katabumarking.append(center_katabummarking[1])
        
        katabupitchresult = check_tolerance(katabumarking_lengths, pitchSpecKatabu, pitchToleranceKatabu)


        xy_pairs_katabumarking = list(zip(detectedposX_katabumarking, detectedposY_katabumarking))
        draw_pitch_line(img_katabumarking, xy_pairs_katabumarking, katabupitchresult, endoffset_y, thickness=2)

        #pick only the first element if array consists of more than 1 element
        if len(katabumarking_lengths) > 1:
            katabumarking_lengths = katabumarking_lengths[:1]
        if len(katabupitchresult) > 1:
            katabupitchresult = katabupitchresult[:1]
        #since there is only one katabu marking, we can just use the first element
        if katabumarking_lengths:
            deltaPitchKatabu = [katabumarking_lengths[0] - pitchSpecKatabu[0]]
        else:
            deltaPitchKatabu = [0]


    for i, detection in enumerate(sorted_detections):
        
        bbox = detection.bbox
        x, y = get_center(bbox)
        w = bbox.maxx - bbox.minx
        h = bbox.maxy - bbox.miny
        class_id = detection.category.id
        class_name = detection.category.name
        score = detection.score.value

        detectedid.append(class_id)
        detectedposX.append(x)
        detectedposY.append(y)

        #id 0 object is white clip
        #id 1 object is brown clip
        #id 2 object is yellow clip
        #id 3 object is orange clip

        if partid == "LH":
            if i < len(clipSpecLH) and class_id == clipSpecLH[i]:
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)
        if partid == "RH":
            if i < len(clipSpecRH) and class_id == clipSpecRH[i]:
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)

        center = draw_bounding_box(img, x, y, w, h, [img.shape[1], img.shape[0]], color=color)

        ## Made this for viz only
        # if class_id == 0: #Clip is white
        #     center = draw_bounding_box(img, x, y, w, h, [img.shape[1], img.shape[0]], color=(255, 255, 255))
        # if class_id == 1: #Clip is brown
        #     center = draw_bounding_box(img, x, y, w, h, [img.shape[1], img.shape[0]], color=(139, 69, 19))
        # if class_id == 2: #Clip is yellow
        #     center = draw_bounding_box(img, x, y, w, h, [img.shape[1], img.shape[0]], color=(255, 255, 0))
        # if class_id == 3: #Clip is orange
        #     center = draw_bounding_box(img, x, y, w, h, [img.shape[1], img.shape[0]], color=(255, 165, 0))

        if prev_center is not None:
            length = calclength(prev_center, center)*pixelMultiplier
            middle_lengths.append(length)
            line_center = ((prev_center[0] + center[0]) // 2, (prev_center[1] + center[1]) // 2)
            if i != 1 and i != len(sorted_detections) - 1:
                img = drawbox(img, line_center, length)
                img = drawtext(img, line_center, length)
        prev_center = center

    #not used temporarily

    # for detect_custom in hanire_detection:
    #     class_id_custom, x_custom, y_custom, _, _, _ = detect_custom
    #     class_id_custom = int(class_id_custom)
    #     customid.append(class_id_custom)

    #     if class_id_custom == 0:
    #         drawcircle(img, (x_custom*img.shape[1], y_custom*img.shape[0]), 0)
    #     elif class_id_custom == 1:
    #         drawcircle(img, (x_custom*img.shape[1], y_custom*img.shape[0]), 1)
        

    detectedPitch = middle_lengths

    #pop first and last element of the list
    checkedPitchResult = detectedPitch[1:-1]
    detectedposX = detectedposX[1:-1]
    detectedposY = detectedposY[1:-1]

    if partid == "LH":
        pitchresult = check_tolerance(checkedPitchResult, pitchSpecLH, pitchToleranceLH)

        if len(checkedPitchResult) == 7:
            deltaPitch = [checkedPitchResult[i] - pitchSpecLH[i] for i in range(len(pitchSpecLH))]
        else:
            deltaPitch = [0, 0, 0, 0, 0, 0, 0]
            checkedPitchResult = [0, 0, 0, 0, 0, 0, 0]

        allpitchresult = checkedPitchResult + katabumarking_lengths #weird naming, this is a list of all the clip pitch and the katabu marking pitch
        pitchresult = pitchresult + katabupitchresult #also weird naming, this is a list of 0 and 1 value for whether the tolerance is fullfilled
        deltaPitch = deltaPitch + deltaPitchKatabu #this is the delta (difference) between the nominal pitch and the detected pitch

        if any(result != 1 for result in pitchresult):
            flag_pitchfuryou = 1
        #check whether the detectedid matches with the clipSpecLH
        if detectedid != clipSpecLH:
            flag_clip_furyou = 1

        if flag_clip_furyou or flag_clip_hanire or flag_pitchfuryou:
            status = "NG"
        else:
            status = "OK"

        # print(pitchresult)

    if partid == "RH":
        pitchresult = check_tolerance(checkedPitchResult, pitchSpecRH, pitchToleranceRH)

        if len(checkedPitchResult) == 7:
            deltaPitch = [checkedPitchResult[i] - pitchSpecRH[i] for i in range(len(pitchSpecRH))]
        else:
            deltaPitch = [0, 0, 0, 0, 0, 0, 0]
            checkedPitchResult = [0, 0, 0, 0, 0, 0, 0]

        allpitchresult = checkedPitchResult + katabumarking_lengths #weird naming, this is a list of all the clip pitch and the katabu marking pitch
        pitchresult = pitchresult + katabupitchresult #also weird naming, this is a list of 0 and 1 value for whether the tolerance is fullfilled
        deltaPitch = deltaPitch + deltaPitchKatabu #this is the delta (difference) between the nominal pitch and the detected pitch

        if any(result != 1 for result in pitchresult):
            flag_pitchfuryou = 1
        #check whether the detectedid matches with the clipSpecLH
        # print (f"spec {clipSpecRH} detected {detectedid}")
        if detectedid != clipSpecRH:
            flag_clip_furyou = 1

        if flag_clip_furyou or flag_clip_hanire or flag_pitchfuryou:
            status = "NG"
        else:
            status = "OK"

        print(pitchresult)




    xy_pairs = list(zip(detectedposX, detectedposY))
    draw_pitch_line(img, xy_pairs, pitchresult, endoffset_y)

    play_sound(status)
    img = draw_status_text(img, status)

    img = draw_flag_status(img, flag_pitchfuryou, flag_clip_furyou, flag_clip_hanire)


    return img, img_katabumarking, allpitchresult, pitchresult, deltaPitch, flag_clip_hanire, status



def dailytenkencheck(img, img_katabumarking, detections, katabumarking_detection, hanire_detection, partid=None):

    sorted_detections = sorted(detections, key=lambda d: d.bbox.minx)


    middle_lengths = []
    katabumarking_lengths = []

    detectedid = []
    customid = []

    detectedPitch = []
    deltaPitch = []
    deltaPitchKatabu = []

    detectedposX = []
    detectedposY = []


    detectedposX_katabumarking = []
    detectedposY_katabumarking = []

    pitchresult = []
    checkedPitchResult = []

    katabupitchresult = []

    allpitchresult = []

    prev_center = None
    prev_center_katabumarking = None

    flag_pitchfuryou = 0
    flag_clip_furyou = 0
    flag_clip_hanire = 0

    #KATABU MARKING DETECTION
    #class 0 is for clip, class 1 is for katabu marking
    if partid == "tenken03":
        status = "NG"
        for r in katabumarking_detection:
            for box in r.boxes:
                x_marking, y_marking = float(box.xywh[0][0].cpu()), float(box.xywh[0][1].cpu())
                w_marking, h_marking = float(box.xywh[0][2].cpu()), float(box.xywh[0][3].cpu())
                class_id_marking = int(box.cls.cpu())

                if class_id_marking == 0:
                    color = (0, 255, 0)
                elif class_id_marking == 1:
                    color = (100, 100, 200)

                center_katabumarking = draw_bounding_box(img_katabumarking, 
                                        x_marking, y_marking, 
                                        w_marking, h_marking, 
                                        [img_katabumarking.shape[1], img_katabumarking.shape[0]], color=color,
                                        bbox_offset=3, thickness=2)
                
                if x_marking is not None:
                    status = "OK"
            
            katabupitchresult = check_tolerance(katabumarking_lengths, pitchSpecKatabu, pitchToleranceKatabu)


            xy_pairs_katabumarking = list(zip(detectedposX_katabumarking, detectedposY_katabumarking))
            draw_pitch_line(img_katabumarking, xy_pairs_katabumarking, katabupitchresult, endoffset_y, thickness=2)

            #pick only the first element if array consists of more than 1 element
            if len(katabumarking_lengths) > 1:
                katabumarking_lengths = katabumarking_lengths[:1]
            if len(katabupitchresult) > 1:
                katabupitchresult = katabupitchresult[:1]
            #since there is only one katabu marking, we can just use the first element
            if katabumarking_lengths:
                deltaPitchKatabu = [katabumarking_lengths[0] - pitchSpecKatabu[0]]
            else:
                deltaPitchKatabu = [0]
        
        img = draw_status_text(img, status)
        img_katabumarking = draw_status_text(img_katabumarking, status, size="small")
        # cv2.imwrite("img_katabumarking.jpg", img_katabumarking)

    if partid == "tenken01":

        for i, detection in enumerate(sorted_detections):
            
            bbox = detection.bbox
            x, y = get_center(bbox)
            w = bbox.maxx - bbox.minx
            h = bbox.maxy - bbox.miny
            class_id = detection.category.id
            class_name = detection.category.name
            score = detection.score.value

            detectedid.append(class_id)
            detectedposX.append(x)
            detectedposY.append(y)

            if i < len(dailytenken01ClipSpec) and class_id == dailytenken01ClipSpec[i]:
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)

            center = draw_bounding_box(img, x, y, w, h, [img.shape[1], img.shape[0]], color=color)

            if prev_center is not None:
                length = calclength(prev_center, center)*pixelMultiplier
                middle_lengths.append(length)
                line_center = ((prev_center[0] + center[0]) // 2, (prev_center[1] + center[1]) // 2)
                img = drawbox(img, line_center, length)
                img = drawtext(img, line_center, length)
            prev_center = center


        detectedPitch = middle_lengths
        checkedPitchResult = detectedPitch
        detectedposX = detectedposX
        detectedposY = detectedposY

        pitchresult = check_tolerance(checkedPitchResult, dailytenken01Spec, dailytenken01Tolerance)

        if len(checkedPitchResult) == 5:
            deltaPitch = [checkedPitchResult[i] - dailytenken01Spec[i] for i in range(len(dailytenken01Spec))]
        else:
            deltaPitch = [0, 0, 0, 0, 0]
            checkedPitchResult = [0, 0, 0, 0, 0]

        allpitchresult = checkedPitchResult #weird naming, this is a list of all the clip pitch and the katabu marking pitch
        pitchresult = pitchresult #also weird naming, this is a list of 0 and 1 value for whether the tolerance is fullfilled
        deltaPitch = deltaPitch #this is the delta (difference) between the nominal pitch and the detected pitch

        if any(result != 1 for result in pitchresult):
            flag_pitchfuryou = 1

        #check whether the detectedid matches with the tenken01
        if detectedid != dailytenken01ClipSpec:
            flag_clip_furyou = 1

        if flag_clip_furyou or flag_clip_hanire or flag_pitchfuryou:
            status = "NG"
        else:
            status = "OK"

        xy_pairs = list(zip(detectedposX, detectedposY))
        draw_pitch_line(img, xy_pairs, pitchresult, endoffset_y)

        play_sound(status)
        img = draw_status_text(img, status)
        img_katabumarking = np.zeros((160, 320, 3), dtype=np.uint8)

    if partid == "tenken02":
        
        for i, detection in enumerate(sorted_detections):
            
            bbox = detection.bbox
            x, y = get_center(bbox)
            w = bbox.maxx - bbox.minx
            h = bbox.maxy - bbox.miny
            class_id = detection.category.id
            class_name = detection.category.name
            score = detection.score.value

            detectedid.append(class_id)
            detectedposX.append(x)
            detectedposY.append(y)

            if i < len(dailytenken02ClipSpec) and class_id == dailytenken02ClipSpec[i]:
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)

            center = draw_bounding_box(img, x, y, w, h, [img.shape[1], img.shape[0]], color=color)

        #check whether the detectedid matches with the tenken01
        if detectedid != dailytenken02ClipSpec:
            flag_clip_furyou = 1

        if flag_clip_furyou or flag_clip_hanire or flag_pitchfuryou:
            status = "NG"
        else:
            status = "OK"

        play_sound(status)
        img = draw_status_text(img, status)
        img_katabumarking = np.zeros((160, 320, 3), dtype=np.uint8)

    return img, img_katabumarking, allpitchresult, pitchresult, deltaPitch, flag_clip_hanire, status


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


def draw_pitch_line(image, xy_pairs, pitchresult, endoffset_y=0, thickness=4):
    xy_pairs = [(int(x), int(y)) for x, y in xy_pairs]

    if len(xy_pairs) != 0:
        for i in range(len(xy_pairs) - 1):
            if i < len(pitchresult) and pitchresult[i] is not None:
                if pitchresult[i] == 1:
                    lineColor = (0, 255, 0)
                else:
                    lineColor = (255, 0, 0)

            # if i == 0:
            #     offsetpos_ = (xy_pairs[i+1][0], xy_pairs[i+1][1] + endoffset_y)
            #     cv2.line(image, xy_pairs[i], offsetpos_, lineColor, 5)
            #     cv2.circle(image, xy_pairs[i], 4, (255, 0, 0), -1)
            # elif i == len(xy_pairs) - 2:
            #     offsetpos_ = (xy_pairs[i][0], xy_pairs[i][1] + endoffset_y)
            #     cv2.line(image, offsetpos_, xy_pairs[i+1], lineColor, 5)
            #     cv2.circle(image, xy_pairs[i+1], 4, (255, 0, 0), -1)
            # else:
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
    # print(checkedPitchResult, result)
    return result

def yolo_to_pixel(yolo_coords, img_shape):
    class_id, x, y, w, h, confidence = yolo_coords
    x_pixel = int(x * img_shape[1])
    y_pixel = int(y * img_shape[0])
    return x_pixel, y_pixel

def find_edge_point(image, center, direction="None", offsetval = 0):
    x, y = center[0], center[1]
    blur = 0
    brightness = 0
    contrast = 1
    lower_canny = 100
    upper_canny = 200

    #read canny value from /aikensa/param/canyparams.yaml if exist
    if os.path.exists("./aikensa/param/cannyparams.yaml"):
        with open("./aikensa/param/cannyparams.yaml") as f:
            cannyparams = yaml.load(f, Loader=yaml.FullLoader)
            blur = cannyparams["blur"]
            brightness = cannyparams["brightness"]
            contrast = cannyparams["contrast"]
            lower_canny = cannyparams["lower_canny"]
            upper_canny = cannyparams["upper_canny"]

    # Apply adjustments
    adjusted_image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
    gray_image = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (blur | 1, blur | 1), 0)
    canny_img = cv2.Canny(blurred_image, lower_canny, upper_canny)

    # cv2.imwrite(f"adjusted_image_{direction}.jpg", adjusted_image)
    # cv2.imwrite(f"gray_image.jpg_{direction}", gray_image)
    # cv2.imwrite(f"blurred_image.jpg_{direction}", blurred_image)
    # cv2.imwrite(f"canny_debug.jpg_{direction}", canny_img)


    while 0 <= x < image.shape[1]:
        if canny_img[y + offsetval, x] == 255:  # Found an edge
            # cv2.line(image, (center[0], center[1] + offsetval), (x, y + offsetval), (0, 255, 0), 1)
            # color = (0, 0, 255) if direction == "left" else (255, 0, 0)
            # cv2.circle(image, (x, y + offsetval), 5, color, -1)
            return x, y + offsetval
        
        x = x - 1 if direction == "left" else x + 1
    return None

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

class BoundingBox:
    def __init__(self, minx, miny, maxx, maxy):
        self.minx = minx
        self.miny = miny
        self.maxx = maxx
        self.maxy = maxy

class PredictionScore:
    def __init__(self, value):
        self.value = value

class Category:
    def __init__(self, id, name):
        self.id = id
        self.name = name

class ObjectPrediction:
    def __init__(self, bbox, score, category):
        self.bbox = bbox
        self.score = score
        self.category = category