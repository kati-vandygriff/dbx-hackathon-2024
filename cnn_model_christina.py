pip install opencv-python opencv-python-headless tensorflow

import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Function to parse a single XML file
# got data from https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection/data
def parse_voc_xml(xml_file): # function for indificual file 
  tree = ET.parse(xml_file) # get tree and root of XML tree
  root = tree.getroot()
    
  helmets = [] # initialize list
  filename = root.find('filename').text # get file name 
  for obj in root.findall('object'): # iterates through and gets bounding box 
    name = obj.find('name').text
    if name == 'helmet':  # Only keep the helmet class
      bndbox = obj.find('bndbox')
      xmin = int(bndbox.find('xmin').text)
      ymin = int(bndbox.find('ymin').text)
      xmax = int(bndbox.find('xmax').text)
      ymax = int(bndbox.find('ymax').text)
      helmets.append({ # append as dictionary format 
        'filename': filename,
        'bbox': [xmin, ymin, xmax, ymax]
      })
    
  return helmets

# Function to parse all XML files in the directory
def parse_all_voc_xml(annotation_dir): 
  all_helmets = [] # initialize list 
  for xml_file in os.listdir(annotation_dir):
    if xml_file.endswith('.xml'): # makes sure file is xml 
      xml_path = os.path.join(annotation_dir, xml_file)
      helmet_bboxes = parse_voc_xml(xml_path) # process individual file 
      all_helmets.extend(helmet_bboxes)
    
  return all_helmets

 # defining the helmets 
class HelmetDataGenerator(Sequence):
  def __init__(self, annotations, image_dir, batch_size=32, input_size=(224, 224), augment=True):
    self.annotations = annotations # annotations for each image
    self.image_dir = image_dir # image directory
    self.batch_size = batch_size # number of smaples per batch 
    self.input_size = input_size # desired image of each image 
    self.augment = augment # boolean flag whether to apply data augmentation
    self.indexes = np.arange(len(self.annotations)) # array of indices for dataset 
  
    self.datagen = ImageDataGenerator( # initiatilizes objext from Keras to apply data augmentations
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )
        
  def __len__(self):
    return int(np.floor(len(self.annotations) / self.batch_size)) # returns numb of bathces per epoch 
    
  def __getitem__(self, index): # retrieves the indices for current batch and extracts annotations
    batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
    batch_annotations = [self.annotations[i] for i in batch_indexes]

    # initialize lists t store     
    images = []
    labels = []
        
    for annotation in batch_annotations: # load images using OpenCV
      image_path = os.path.join(self.image_dir, annotation['filename'])
      image = cv2.imread(image_path)
            
      if image is None:
        continue  # Skip this image if it cannot be loaded

      # converts image and resizes it - gets bounding box and extracts it    
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
      xmin, ymin, xmax, ymax = annotation['bbox']
      helmet_img = image[ymin:ymax, xmin:xmax]
      helmet_img = cv2.resize(helmet_img, self.input_size)
            
      images.append(helmet_img)
      labels.append(1)  # Label for helmet is 1 (binary classification)

    if len(images) == 0:
      # Handle cases where no valid images were loaded
      return np.zeros((self.batch_size, *self.input_size, 3)), np.zeros(self.batch_size)

    # converts lists to arrays     
    images = np.array(images)
    labels = np.array(labels)

    if self.augment:
      images = next(self.datagen.flow(images, batch_size=self.batch_size, shuffle=False))
        
    return images, labels
    
  def on_epoch_end(self): # shuffels indices after each epoch to ensure data is not processed in same other each time, improces model
    np.random.shuffle(self.indexes)

 

# define cnn model -> need to mess around with inputs, just did a simple one 
def create_cnn_model(input_shape=(224, 224, 3)):
  model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Binary classification (helmet vs no helmet)
  ])
    
  model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

  return model

annotation_dir = "path_to_annotation_dir" # directory with xml files 
jpg_dir = "path_to_jpg_dir" # directory with jpg 
  
# Load and split the annotations
annotations = parse_all_voc_xml(annotation_dir)
train_annotations, val_annotations = train_test_split(annotations, test_size=0.2, random_state=42)

# train and validation generators 
train_generator = HelmetDataGenerator(train_annotations, jpg_dir, batch_size=32, input_size=(224, 224))
val_generator = HelmetDataGenerator(val_annotations, jpg_dir, batch_size=32, input_size=(224, 224), augment=False)

# create model 
model = create_cnn_model(input_shape=(224, 224, 3))

# train model
model.fit(train_generator, validation_data=val_generator, epochs=6, steps_per_epoch=len(train_generator), validation_steps=len(val_generator))

# need to add test_xml but just using orig now 
test_annotations = parse_all_voc_xml(annotation_dir)

# same with jpg 
test_generator = HelmetDataGenerator(test_annotations, jpg_dir, batch_size=32, input_size=(224, 224), augment=False)

from PIL import Image
has_helmet_picture = "path_to_single_helmet_jpg"
no_helmet_picture = "path_to_single_no_helmet_jpg"

def preprocess_image(image_path, input_size=(224, 224)):
  image = cv2.imread(image_path)

  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image = cv2.resize(image, input_size)
  image = image.astype('float32')/255.0
  image = np.expand_dims(image, axis=0)

  return image

def make_prediction(image):
  resized_image = preprocess_image(image)
  prediction = model.predict(resized_image)
  answer = "Helmet" if prediction == 1.0 else "Missing Helmet"
  return answer, prediction

