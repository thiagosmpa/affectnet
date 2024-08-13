import numpy as np
import cv2
import math
from PIL import Image, ImageDraw, ImageFont

def norm_coordinates(normalized_x, normalized_y, image_width, image_height):
    
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    
    return x_px, y_px

def get_box(fl, w, h):
    idx_to_coors = {}
    for idx, landmark in enumerate(fl.landmark):
        landmark_px = norm_coordinates(landmark.x, landmark.y, w, h)

        if landmark_px:
            idx_to_coors[idx] = landmark_px

    x_min = np.min(np.asarray(list(idx_to_coors.values()))[:,0])
    y_min = np.min(np.asarray(list(idx_to_coors.values()))[:,1])
    endX = np.max(np.asarray(list(idx_to_coors.values()))[:,0])
    endY = np.max(np.asarray(list(idx_to_coors.values()))[:,1])

    (startX, startY) = (max(0, x_min), max(0, y_min))
    (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
    
    return startX, startY, endX, endY

def display_FPS(img, text, margin=1.0, box_scale=1.0):
    img_h, img_w, _ = img.shape
    line_width = int(min(img_h, img_w) * 0.001) 
    thickness = max(int(line_width / 3), 1)

    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_color = (0, 0, 0)
    font_scale = thickness / 1.5

    t_w, t_h = cv2.getTextSize(text, font_face, font_scale, None)[0]

    margin_n = int(t_h * margin)
    
    bottom_left_x = img_w - t_w - margin_n - int(2 * t_h * box_scale)
    bottom_left_y = img_h - margin_n - int(2 * t_h * box_scale) - t_h

    sub_img = img[bottom_left_y: bottom_left_y + t_h + int(2 * t_h * box_scale),
                  bottom_left_x: img_w - margin_n]

    white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255

    img[bottom_left_y: bottom_left_y + t_h + int(2 * t_h * box_scale),
        bottom_left_x: img_w - margin_n] = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)

    cv2.putText(img=img,
                text=text,
                org=(bottom_left_x + int(t_h * box_scale) // 2,
                     bottom_left_y + t_h + int(t_h * box_scale) // 2),
                fontFace=font_face,
                fontScale=font_scale,
                color=font_color,
                thickness=thickness,
                lineType=cv2.LINE_AA,
                bottomLeftOrigin=False)

    return img

def drawBox(frame, box, color=(0, 0, 255), thickness=3, alpha=0.08):
    x1, y1, x2, y2 = box
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame_pil)

    # overlay = Image.new('RGBA', frame_pil.size, (0, 0, 0, 0))
    # draw_overlay = ImageDraw.Draw(overlay)
    # draw_overlay.rectangle([x1, y1, x2, y2], fill=(0, 0, 255, int(255 * alpha)))
    # frame_pil = Image.alpha_composite(frame_pil.convert('RGBA'), overlay).convert('RGB')

    draw = ImageDraw.Draw(frame_pil)
    draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=thickness)

    frame[:] = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

def drawText(frame, text, position, text_color=(255, 255, 255), font_scale=1.2, font_thickness=1, margin=10, border_radius=10):
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame_pil)
    
    try:
        font = ImageFont.truetype("../font/JetBrainsMonoNL-Regular.ttf", int(font_scale * 15))
    except IOError:
        font = ImageFont.load_default()

    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    text_x, text_y = position
    text_x = min(text_x + 1, frame.shape[1] - text_w - margin)
    text_y = position[1]

    box_coords = [text_x - margin, text_y - text_h, text_x + text_w + margin, text_y + margin + text_h]
    draw.rounded_rectangle(box_coords, radius=border_radius, fill=(0, 0, 255))
    draw.text((text_x, text_y+2 - text_h), text, font=font, fill=text_color)

    frame[:] = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

def drawTitle(frame, title, font_scale=2, margin=15, margin_top=None, border_radius=10, fill=(255, 255, 255), alpha=0.8):
    try:
        font = ImageFont.truetype("../font/JetBrainsMonoNL-Regular.ttf", int(font_scale * 15))
    except IOError:
        font = ImageFont.load_default()

    frame_w = frame.shape[1]
    frame_h = frame.shape[0]

    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay = Image.new('RGBA', frame_pil.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    
    title_bbox = draw_overlay.textbbox((0, 0), title, font=font)
    title_w = title_bbox[2] - title_bbox[0]
    title_h = title_bbox[3] - title_bbox[1]
    
    title_x = (frame_w - title_w) // 2
    title_y = margin if margin_top is None else margin_top

    box_coords = [title_x - margin, title_y, title_x + title_w + margin, title_y + title_h + margin]
    
    fill_with_alpha = fill + (int(255 * alpha),)
    draw_overlay.rounded_rectangle(box_coords, radius=border_radius, fill=fill_with_alpha)
    
    combined = Image.alpha_composite(frame_pil, overlay)
    
    combined = combined.convert("RGB")
    titleDraw = ImageDraw.Draw(combined)
    titleDraw.text((title_x, title_y), title, font=font, fill=(0, 0, 0))
    
    frame[:] = cv2.cvtColor(np.array(combined), cv2.COLOR_RGB2BGR)

def annotate(frame, box, label):
    drawBox(frame, box)
    x1, y1, _, _ = box
    drawText(frame, label, (box[2] + 10, y1 + 10), font_scale=2.5)
    # drawTitle(frame, title, margin_top=20)
    return frame