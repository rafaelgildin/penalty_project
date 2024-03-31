from IPython.display import display
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO
import cv2 as cv
import keyboard # get the keys typed
import os, serial, time

COLOR_BLACK = (0, 0, 0)
COLOR_RED = (255, 0, 0)
COLOR_BLUE = (0, 0, 255)

def red_led_on(ser): ser.write(b'R')
def red_led_off(ser): ser.write(b'r')
def green_led_on(ser): ser.write(b'G')
def green_led_off(ser): ser.write(b'g')
def yellow_led_on(ser): ser.write(b'Y')
def yellow_led_off(ser): ser.write(b'y')
def blink(ser):
    print("yes detected")
    red_led_on(ser)
    green_led_on(ser)
    yellow_led_on(ser)
    time.sleep(2)
    red_led_off(ser)
    green_led_off(ser)
    yellow_led_off(ser)

def get_cap_props(cap):
    cap_props = {}
    cap_props['fps'] = int(cap.get(cv.CAP_PROP_FPS))
    cap_props['width'] = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    cap_props['height'] = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    cap_props['frame_count'] = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    cap_props['total_duration'] = cap_props['frame_count'] / cap_props['fps']
    print(f"Video (sec): {0}-{round(cap_props['total_duration'],2)}")
    print(f"Frames: {0}-{cap_props['frame_count']}")
    return cap_props

def plot_frame(frame,width,height,text,font_color=COLOR_BLACK):
    # Define the position and size of the rectangle
    top_left_corner = (0, 0)
    bottom_right_corner = (width, 100)
    rectangle_color = (255, 255, 255)  # White color in BGR
    rectangle_thickness = -1  # Filled rectangle

    # Draw the white rectangle
    cv.rectangle(frame, top_left_corner, bottom_right_corner, rectangle_color, rectangle_thickness)

    # Write text on the frame using OpenCV before converting it to RGB
    font = cv.FONT_HERSHEY_SIMPLEX
    position = (50, 70)
    font_scale = 0.5
    line_type = 2

    # Calculate text size
    (text_width, text_height), _ = cv.getTextSize(text, font, font_scale, line_type)    

    # Adjust font scale if text width exceeds the rectangle width
    while text_width > width - 20:  # Subtracting 20 for some padding
        font_scale -= 0.1  # Decrease font scale
        (text_width, text_height), _ = cv.getTextSize(text, font, font_scale, line_type)
        if font_scale <= 0.1:  # Prevent font scale from becoming too small or negative
            break

    # Calculate the center position
    text_x = (width - text_width) // 2
    text_y = top_left_corner[1] + ((bottom_right_corner[1] - top_left_corner[1]) + text_height) // 2

    # The position where the text will start
    position = (text_x, text_y)

    # Put the text on the frame
    cv.putText(frame, text, position, font, font_scale, font_color, line_type)
    cv.imshow('Webcam', frame)
    
def plot_frame_notebook(frame,width,height,text,font_color=COLOR_BLACK):
    # Define the position and size of the rectangle
    top_left_corner = (0, 0)
    bottom_right_corner = (width, 100)
    rectangle_color = (255, 255, 255)  # White color in BGR
    rectangle_thickness = -1  # Filled rectangle

    # Draw the white rectangle
    cv.rectangle(frame, top_left_corner, bottom_right_corner, rectangle_color, rectangle_thickness)

    # Write text on the frame using OpenCV before converting it to RGB
    font = cv.FONT_HERSHEY_SIMPLEX
    position = (50, 70)
    font_scale = 1.5
    line_type = 2

    # Calculate text size
    (text_width, text_height), _ = cv.getTextSize(text, font, font_scale, line_type)    

    # Adjust font scale if text width exceeds the rectangle width
    while text_width > width - 20:  # Subtracting 20 for some padding
        font_scale -= 0.1  # Decrease font scale
        (text_width, text_height), _ = cv.getTextSize(text, font, font_scale, line_type)
        if font_scale <= 0.1:  # Prevent font scale from becoming too small or negative
            break

    # Calculate the center position
    text_x = (width - text_width) // 2
    text_y = top_left_corner[1] + ((bottom_right_corner[1] - top_left_corner[1]) + text_height) // 2

    # The position where the text will start
    position = (text_x, text_y)
    # Put the text on the frame
    cv.putText(frame, text, position, font, font_scale, font_color, line_type)
    plt.figure(figsize=(10, 5))
    plt.imshow(frame)
    plt.axis('off')
    plt.show()