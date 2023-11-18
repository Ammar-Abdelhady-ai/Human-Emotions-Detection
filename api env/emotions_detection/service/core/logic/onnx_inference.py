
import onnxruntime as rt
import cv2 as cv
import numpy as np
import time
import service.main as s



def emotions_detector(img_array):
    
    if len(img_array.shape) == 2:
        img_array = cv.cvtColor(img_array, cv.Color_GRAY2RGB)
        
    time_init = time.time()
    
    
    # providers = ["CPUExecutionProvider"] # ["CPUExecutionProvider", "CUDAExecutionProvider"]
    # output_path = r"service\vit_keras.onnx"
    # m_q = rt.InferenceSession(output_path, providers=providers)
    


    test_image = cv.resize(img_array, (256, 256))
    im = test_image.astype(np.float32)
    img_array = np.expand_dims(im, axis=0)

    time_elapsed_preprocess= time.time() - time_init    


    input_data = {"input": img_array}
    onnx_pred = s.m_q.run(["dense"], input_data) 
    
    time_elapsed = time.time() - time_init

    pred = np.argmax(onnx_pred[0][0])
    print(pred)

    emotion = ""

    if pred == 0:
        emotion = "Angry"

    elif pred == 1:
        emotion = "Happy"
        
    else:
        emotion = "Sad"

    return {
        "emotion": emotion, 
        "time_elapsed": str(time_elapsed), 
        "time_elapsed_preprocess": str(time_elapsed_preprocess), 
        }