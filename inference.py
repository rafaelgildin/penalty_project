from utils import *
    
    
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
    
def video_inference(with_arduino=False):
    video_path = os.path.join('data','videos','light_left_feet_left_goal_10_goals_ok','lf13.mp4')
    frame_start,frame_end = 10,17
    best_model_path = os.path.join('models', 'best_train_2.pt')
    model = YOLO(best_model_path)
    cap = cv.VideoCapture(video_path)
    
    # Get the video codec and properties from the input video 
    fps = int(cap.get(cv.CAP_PROP_FPS))
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    total_duration = frame_count / fps 
    color = COLOR_BLACK
    print(f"Video (sec): {0}-{round(total_duration,2)}")
    print(f"Frames: {0}-{frame_count}")
    i = 0
    # frame_number = 0
    # while(True):
    
    for i in range(frame_start,frame_end):
        cap.set(cv.CAP_PROP_POS_FRAMES, i)
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
        # frame_number+=1
        # i+=1
        classes_probs_text = f"No ({round((classes_probs['no']*100.0),2)}%) - Yes ({round((classes_probs['yes']*100.0),2)}%)" 
        text = f"Frame {i}/{frame_count} - Time {round(frame_time,5)} ms - {classes_probs_text}"
        if (classes_probs['yes'] > 0.5): 
            color=COLOR_BLUE # red in practice
            if(with_arduino): green_led_on()
        plot_frame(frame,width,height,text,color)
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
    green_led_off()
    video_inference(with_arduino=True)
    ser.close()
    print('end of inference')
    
video_inference()