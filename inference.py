from utils import *

def webcam_inference():
    best_model_path = os.path.join('models', 'best_train_2.pt')
    # base_model_path = 'yolov8n.pt'
    color = COLOR_BLACK
    model = YOLO(best_model_path)
    cap = cv.VideoCapture(0)
    cap_props = get_cap_props(cap)
    frame_number = 0
    i = 0
    green_led_on(ser)


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
        text = f"Frame {frame_number}/{cap_props['frame_count']} - Time {round(frame_time,5)} ms - {classes_probs_text}"
        text = f"{classes_probs_text}"
        plot_frame(frame,cap_props['width'],cap_props['height'],text,color)
        
        # # if(i >= 50): # condition based on counter
        if (classes_probs['yes'] > 0.8): # condition based on prob
            color=COLOR_BLUE # red in practice
            # i = 0
            # blink(ser)
            # while True:
            #     print("Looping... Press any key to stop.")
            #     if keyboard.read_event(suppress=True).event_type == keyboard.KEY_DOWN:
            #         print("Key pressed, stopping loop.")
            #         break
        else:
            color=COLOR_BLACK
        if cv.waitKey(1) == ord('q'): break
    cap.release()
    cv.destroyAllWindows()
    
def video_inference(with_arduino=False):
    video_path = os.path.join('data','videos','light_left_feet_left_goal_10_goals_ok','lf13.mp4')
    frame_start,frame_end = 10,17
    best_model_path = os.path.join('models', 'best_train_2.pt')
    model = YOLO(best_model_path)
    cap = cv.VideoCapture(video_path)
    cap_props = get_cap_props(cap)
    color = COLOR_BLACK
    frame_number = -1

    while(True):
        frame_number += 1
    # for frame_number in range(frame_start,frame_end):
        cap.set(cv.CAP_PROP_POS_FRAMES, frame_number)
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
        classes_probs_text = f"No ({round((classes_probs['no']*100.0),2)}%) - Yes ({round((classes_probs['yes']*100.0),2)}%)" 
        text = f"Frame {frame_number}/{cap_props['frame_count']} - Time {round(frame_time,5)} ms - {classes_probs_text}"
        if (classes_probs['yes'] > 0.5): 
            color=COLOR_BLUE # red in practice
            if(with_arduino): red_led_on(ser)
        else:
            color=COLOR_BLACK
            if(with_arduino): red_led_off(ser)
        plot_frame(frame,cap_props['width'],cap_props['height'],text,color)
        cv.waitKey(500)
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

def arduino_webcam_inference():
    global ser
    ser = serial.Serial('COM3', 9600, timeout=1)
    time.sleep(2) # Wait for the connection to be established
    webcam_inference()
    # blink()
    # time.sleep(2)
    ser.close()
    print('end of inference')
    
def arduino_video_inference():
    global ser
    ser = serial.Serial('COM3', 9600, timeout=1)
    time.sleep(2) # Wait for the connection to be established
    green_led_off(ser)
    video_inference(with_arduino=True)
    ser.close()
    print('end of inference')
    
arduino_webcam_inference()