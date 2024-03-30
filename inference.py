from IPython.display import display
from PIL import Image
from ultralytics import YOLO
import cv2 as cv
import os, serial, time
import keyboard # get the keys typed

def green_led_on(): ser.write(b'G')
def green_led_off(): ser.write(b'g')
def yellow_led_on(): ser.write(b'Y')
def yellow_led_off(): ser.write(b'y')
def blink():
    print("yes detected")
    green_led_off()
    yellow_led_on()
    time.sleep(2)
    yellow_led_off()
    green_led_on()
        
def plot_frame(frame,width,height,text):
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
    font_color = (0, 0, 0)
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
    
    
def webcam_inference():
    best_model_path = os.path.join('models', 'best_train_2.pt')
    # base_model_path = 'yolov8n.pt'
    model = YOLO(best_model_path)
    cap = cv.VideoCapture(0)
    frame_number = 0
    
    # Get the video codec and properties from the input video 
    fps = int(cap.get(cv.CAP_PROP_FPS))
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    total_duration = frame_count / fps 
    print(f"Video (sec): {0}-{round(total_duration,2)}")
    print(f"Frames: {0}-{frame_count}")
    i = 0
    green_led_on()
    
    while True:
        ret, frame= cap.read()
        if not ret: break
        
        r = model(frame,save=False, conf=0.5, iou=0.1,device='0',verbose=False)[0]

        # format frame
        classes_names = r.names
        probs = r.cpu().probs.data.tolist()
        classes_probs = {}
        for k,v in classes_names.items(): classes_probs[v] = round(probs[k],2)
        print("Results : ", classes_probs)
        
        # plot frame
        frame_time = int(cap.get(cv.CAP_PROP_POS_MSEC))
        frame_number+=1
        i+=1
        classes_probs_text = f"No ({round((classes_probs['no']*100.0),2)}%) - Yes ({round((classes_probs['yes']*100.0),2)}%)" 
        text = f"Frame {frame_number}/{frame_count} - Time {round(frame_time,5)} ms - {classes_probs_text}"
        text = f"{classes_probs_text}"
        plot_frame(frame,width,height,text)
        
        # if(i >= 50): # condition based on counter
        if (classes_probs['yes'] > 0.8): # condition based on prob
            i = 0
            blink()
            while True:
                print("Looping... Press any key to stop.")
                if keyboard.read_event(suppress=True).event_type == keyboard.KEY_DOWN:
                    print("Key pressed, stopping loop.")
                    break               
        if cv.waitKey(1) == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()
    
def video_inference():
    video_path = os.path.join('data','videos','light_left_feet_left_goal_10_goals_ok','lf13.mp4')
    frame_start,frame_end = 0,5
    best_model_path = os.path.join('models', 'best_train_2.pt')
    model = YOLO(best_model_path)
    cap = cv.VideoCapture(video_path)
    frame_number = 0
    
    # Get the video codec and properties from the input video 
    fps = int(cap.get(cv.CAP_PROP_FPS))
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    total_duration = frame_count / fps 
    print(f"Video (sec): {0}-{round(total_duration,2)}")
    print(f"Frames: {0}-{frame_count}")
    i = 0
    while(True):
    
    # for i in range(frame_start,frame_end):
        # cap.set(cv.CAP_PROP_POS_FRAMES, i)
        ret, frame= cap.read()
        if not ret: break
        
        r = model(frame,save=False, conf=0.5, iou=0.1,device='0',verbose=False)[0]

        # format frame
        classes_names = r.names
        probs = r.cpu().probs.data.tolist()
        classes_probs = {}
        for k,v in classes_names.items(): classes_probs[v] = round(probs[k],2)
        print("Results : ", classes_probs)
        
        # plot frame
        frame_time = int(cap.get(cv.CAP_PROP_POS_MSEC))
        frame_number+=1
        i+=1
        classes_probs_text = f"No ({round((classes_probs['no']*100.0),2)}%) - Yes ({round((classes_probs['yes']*100.0),2)}%)" 
        text = f"Frame {frame_number}/{frame_count} - Time {round(frame_time,5)} ms - {classes_probs_text}"
        text = f"{classes_probs_text}"
        plot_frame(frame,width,height,text)
        cv.waitKey(0)
    cap.release()
    cv.destroyAllWindows()

def image_inference():
    # change current folder
    # os.chdir('/home/rafaa/pmy/penalty_project/data')
    best_model_path = os.path.join('models', 'best_train_2.pt')
    path = os.path.join('data', 'penalty_kick.jpg')

    # Load the model
    model = YOLO(best_model_path)

    # inference
    r = model(path,save=False, conf=0.5, iou=0.1,device='0')[0]

    # Show the results
    im_array = r.plot(conf=True)  # plot a BGR numpy array of predictions
    # im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    # display(im)
    cv.imshow('image',im_array)
    cv.waitKey(0) 
    
    # Tratando o resultado
    classes_names = r.names
    probs = r.cpu().probs.data.tolist()
    classes_probs = {}
    for k,v in classes_names.items(): classes_probs[v] = round(probs[k],2)
    print("Results : ", classes_probs)
    cv.destroyAllWindows() 

def arduino_inference():
    global ser
    ser = serial.Serial('COM3', 9600, timeout=1)
    time.sleep(2) # Wait for the connection to be established
    webcam_inference()
    # blink()
    # time.sleep(2)
    ser.close()
    print('end of inference')
    
video_inference()