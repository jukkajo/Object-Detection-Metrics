import tensorflow as tf
import numpy as np
import json
import os
import csv
import cv2
import imutils
import sys
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.layers import BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import NASNetMobile
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

# Monitoring for both output heads
early_stopping = EarlyStopping(monitor='val_loss', patience=6, mode='min', restore_best_weights=True)

INIT_LR = 1e-4
BATCH_SIZE = 32
NUM_EPOCHS = 330
BASE_OUTPUT = "Output-Evo-MobileNetV3Snall"
SAVE_NAME = "mobilenetv3s-with-obj-det-head-experimental"

# Paths, adjust accordingly
MODEL_SAVED_PATH = os.path.sep.join([BASE_OUTPUT, SAVE_NAME])
anno_path = "./Boreal/Ruokolahti_annotations/annotations/csv"
image_dir_path= "./Boreal/Ruokolahti_annotations/imgs"

# To display progress bar in terminal
def count_files_with_progress(current_progress, count, notation):
    progress_width = 50
    progress = current_progress / count
    bar_width = int(progress * progress_width)
    progress_bar = f"{notation}: [{'=' * bar_width}{' ' * (progress_width - bar_width)}] {current_progress}/{count}"
    sys.stdout.write('\r' + progress_bar)
    sys.stdout.flush()

train_data = []
train_labels = []
train_annos = []
train_image_paths = []

val_data = []
val_labels = []
val_annos = []
val_image_paths = []

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img_array = np.array(img)
    return img_array

def process_data(count, annotations_path, images_path, notation):
    data = []
    labels = []
    annos = []
    image_paths = []

    iterator = 0
    ii = 0
    if os.path.exists(annotations_path):
        for filename in os.listdir(annotations_path):
            anno_path = os.path.join(annotations_path, filename)
            if os.path.exists(anno_path):
                with open(anno_path, "r") as file:
                    reader_object = csv.reader(file)
                    next(reader_object)

                    for row in reader_object:
                        (label_name, bbox_x, bbox_y, bbox_width, bbox_height, image_name, image_width, image_height) = row
                        
                        bbox_x = int(bbox_x)
                        bbox_y = int(bbox_y)
                        bbox_width = int(bbox_width)
                        bbox_height = int(bbox_height)
                        image_width = int(image_width)
                        image_height = int(image_height)
                         
                        img_path = os.path.sep.join([images_path, (image_name[:-3] + 'jpg')])
                        if os.path.exists(img_path):
                            image_array = preprocess_image(img_path)
                            if (label_name == "Big smoke"):
                                
                                x1 = float(bbox_x) / image_width
                                y1 = float(bbox_y) / image_height
                                x2 = x1 + float(bbox_width) / image_width
                                y2 = y1 + float(bbox_height) / image_height

                                annos.append((x1, y1, x2, y2))
                                data.append(image_array)
                                image_paths.append(img_path)
                                labels.append(0)

                            count_files_with_progress(iterator, count, notation)
                            iterator += 1
            else:
                print("Non-existing annotation path :(")
    else:
        print("Can not find directory of your annotations :(")
    return data, labels, annos, image_paths

# Pre-Processing training data
def count_ds(directory):
    total_rows = 0
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            with open(file_path, mode='r') as file:
                reader = csv.reader(file)
                next(reader, None)
                total_rows += sum(1 for row in reader)
    return total_rows
    
count = count_ds(anno_path)

data, labels, annos, image_paths = process_data(count, anno_path, image_dir_path, "Pre-processing")

data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)
annos = np.array(annos, dtype="float32")
image_paths = np.array(image_paths)

print("Img count:  ", len(data))
print("Label count:  ", len(labels))
print("Anno count:  ", len(annos))
print("Img-path count:  ", len(image_paths))

split = train_test_split(data, labels, annos, image_paths, test_size=0.20, random_state=42)

(train_data, val_data) = split[:2]
(train_labels, val_labels) = split[2:4]
(train_annos, val_annos) = split[4:6]
(train_image_paths, test_image_paths) = split[6:]

#------------------ Architecture and Training: -----------------------

input_tensor = Input(shape=(224, 224, 3))
architecture = MobileNetV3Small(include_top=False, weights='imagenet', input_tensor=input_tensor)
architecture.trainable = False

flatten = Flatten()(architecture.output)

# For Localisation
bounding_box_head = Dense(128, activation="relu")(flatten)
bounding_box_head = Dense(64, activation="relu")(bounding_box_head)
bounding_box_head = Dense(32, activation="relu")(bounding_box_head)
bounding_box_head = Dense(4, activation="sigmoid", name="bounding_box")(bounding_box_head)

# For Classification
softmax_head = Dense(512, activation="relu")(flatten)
softmax_head = Dropout(0.5)(softmax_head)
softmax_head = Dense(512, activation="relu")(softmax_head)
softmax_head = Dropout(0.5)(softmax_head)
softmax_head = Dense(1, activation="sigmoid", name="class_label")(softmax_head)

model = Model(inputs=architecture.input, outputs=[bounding_box_head, softmax_head])

lossWeights = {
    "class_label": 1.0,
    "bounding_box": 1.0
}

opt = Adam(learning_rate=INIT_LR)
model.compile(optimizer=opt, loss={'class_label': 'binary_crossentropy', 'bounding_box': 'mse'}, metrics={'class_label': 'accuracy', 'bounding_box': 'accuracy'})

print(model.summary())

train_targets = {
    "class_label": train_labels,
    "bounding_box": train_annos
}

val_targets = {
    "class_label": val_labels,
    "bounding_box": val_annos
}

print("[INFO] training model...")
H = model.fit(train_data, train_targets, validation_data=(val_data, val_targets), batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, verbose=1, callbacks=[early_stopping])

print("[INFO] saving model in saved_model format...")
tf.saved_model.save(model, MODEL_SAVED_PATH)
