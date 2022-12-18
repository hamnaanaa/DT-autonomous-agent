#!/usr/bin/env python3

from imageai.Detection import ObjectDetection

# instantiating the class  
recognizer = ObjectDetection()

path_model = "./Models/yolo-tiny.h5"
path_input = "./Input/image2.jpg"
path_output = "./Output/newimag2.jpg"

recognizer.setModelTypeAsTinyYOLOv3()
recognizer.setModelPath(path_model)
# loading the pre-trained model TinyYOLOv3
recognizer.loadModel()

# calling the detectObjectsFromImage() function  
recognition = recognizer.detectObjectsFromImage(
    input_image=path_input,
    output_image_path=path_output
)

# iterating through the items found in the image
for eachItem in recognition:
    print(eachItem["name"], " : ", eachItem["percentage_probability"])
