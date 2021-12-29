#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Dec 2021

@author: nina.hosseinikivanani
"""

from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import glob
import cv2
import json
import time
import csv
import ndjson
#import knjson
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm
#%%

key = 'add your azure key'
endpoint = 'add your endpoint'

#%%
computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(key))

#%%
#Open image file

image = [file for file in glob.glob("/IMAGES/*.JPG")]
names = [filename.split("/")[-1]for filename in image]
#display()
#%%
#Example
image_path = "/IMAGES/cat.JPG"
#image_path = "/Users/nina.hosseinikivanan/Desktop/lu/luis_project/sigir_2021/images/*.jpg"
image = open(image, "rb")
# Display the image
display(Image.open(image_path).resize((334, 250))) #In case you need to resize your image
# Call the API

print("===== Description =====")
result_img_des = []
result_des_confidence = []
for i in tqdm(image):
    description_result = computervision_client.describe_image_in_stream(open(i, "rb"))
    # Get the description with confidence level
    #with open('img_des.txt','a')as f:
        #f.write(f"Description:\n")
        #print("Description:")
    if (len(description_result.captions) == 0):
            #f.write(f"No description detected.\n")
        result_img_des.append("NAN")
        result_des_confidence.append("NAN")
    else:
        for caption in description_result.captions:
                #f.write(f"{caption.text} with confidence {caption.confidence * 100:.2f}%\n")
            result_img_des.append(caption.text)
            result_des_confidence.append(caption.confidence)
        time.sleep(5)



df = pd.DataFrame({'Description': result_img_des, 'Confidence': result_des_confidence})
df.to_csv('img_des.csv', index=False,sep = ',',  encoding='utf-8', header= False)


print('Pausing for 60 seconds to avoid triggering rate limit on free account...')
time.sleep (60)
#%%
# Call the API
print("===== Tag =====")
result_img_tag = []
result_tag_confidence = []

for i in image:
    tags_result = computervision_client.tag_image_in_stream(open(i, "rb"))
    #with open("img_tag.csv", 'a')as f:
        #f.write(f"Tags:\n")
        #print("Tags:")
    if (len(tags_result.tags) == 0):
        result_img_tag.append("NAN")
        result_tag_confidence.append("NAN")

            #print("No tags detected.")
            #f.write(f"No tags detected.\n")
    else:
        for tag in tags_result.tags:
            result_img_tag.append(tag.name)
            result_tag_confidence.append(tag.confidence)

                #print(f"{tag.name}: {tag.confidence * 100:.2f}%\n")
                #f.write(f"{tag.name}: {tag.confidence * 100:.2f}%\n")

        time.sleep(5)
        

df = pd.DataFrame({'Tag': result_img_tag, 'Confidence': result_tag_confidence})
df.to_csv('img_tag.csv', index=False, sep =',', encoding='utf-8', header= True)


print('Pausing for 60 seconds to avoid triggering rate limit on free account')
time.sleep (60)
#%%
# Call the API
print("===== Image chategorization =====")
result_img_cat = []
result_cat_confidence = []
for i in image:
    categorize_result = computervision_client.analyze_image_in_stream(open(i, "rb"))
    # Get the categories with confidence score
    #with open('img_cat.txt','a')as f:
        #f.write(f"Categories:\n")
        #print("Categories:")
    if (len(categorize_result.categories) == 0):
            #f.write(f"No categories detected.\n")
        result_img_cat.append("NAN")
        result_cat_confidence.append("NAN")
            #print("No categories detected.\n")
            #f.write(f"No categories detected.\n")
    else:
        for category in categorize_result.categories:
                #f.write(f"{category.name}: {category.score * 100:.2f}%\n")
            result_img_cat.append(category.name)
            result_cat_confidence(category.score)
    time.sleep(5)

                #print(f"{category.name}: {category.score * 100:.2f}%\n")
                
df = pd.DataFrame({'Category': result_img_cat, 'Confidence': result_cat_confidence})
df.to_csv('img_cat.csv', index=False, line_terminator=',', encoding='utf-8', header= True)
print(result_img_cat)
print('Pausing for 60 seconds to avoid triggering rate limit on free account')
time.sleep (60)

#%%
# Call the API
print("===== Detect Objects =====")

# Call API
result_img_obj = []
for i in image[:4]:
    detect_objects_results = computervision_client.detect_objects_in_stream(open(i, "rb"))
    # Print results of detection with bounding boxes
    # Create figure and axes
    #fig, ax = plt.subplots()
    # Display the image
    #ax.imshow(image)
    print("Objects in image:")
    with open('img_obj.txt','a')as f:
        if len(detect_objects_results.objects) == 0:
            result_img_obj.append("No objects detected.")
            print("No objects detected.")
        else:
            for object in detect_objects_results.objects:
                # Create a Rectangle patch
                rect = patches.Rectangle((object.rectangle.x, object.rectangle.y), object.rectangle.w, object.rectangle.h, linewidth=2, edgecolor='r', facecolor='none')
                # Add the patch to the Axes
                tmp = ax.add_patch(rect)
                result_img_obj.append(tmp)
        time.sleep(5)
        #plt.show()
        
print(result_img_obj)

#%%
def dictmaker(filenames, result):
    output_tag = {}
    for key,value in zip(filenames, result):
        output_tag[key]=value
    return(output_tag)

img_des_result = dictmaker(names, result_img_des)
img_tag_result = dictmaker(names, result_img_tag)
img_cat_result = dictmaker(names, result_img_cat)

with open('img_des.json', 'w') as outfile:
    json.dump(img_des_result, outfile)


with open('img_tag.json', 'w') as outfile:
    json.dump(img_tag_result, outfile)

with open('img_cat.json', 'w') as outfile:
    json.dump(img_cat_result, outfile)

with open('img_tag.json', 'r') as outfile:
    print(json.load(outfile))
    





