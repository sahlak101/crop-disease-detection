# crop-disease-detection
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import operator
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array, load_img, array_to_img, ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.image import rgb_to_grayscale
import os
base_path= '/data/govind/project'
os.listdir(base_path)
train_data ='/data/govind/project'
valid_data ='/data/govind/project'
batch_size = 64
image_size = (64, 64)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
    train_data,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical')
valid_datagen = ImageDataGenerator(rescale=1./255)

valid_generator = valid_datagen.flow_from_directory(
    valid_data,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical')
Apple__Apple_scab = os.listdir(r"/data/govind/project/Apple__Apple_scab")
Apple__Black_rot = os.listdir(r"/data/govind/project/Apple__Black_rot")
Apple__Cedar_apple_rust =os.listdir(r"/data/govind/project/Apple__Cedar_apple_rust")
Apple__healthy = os.listdir(r"/data/govind/project/Apple__healthy")
Background_without_leaves = os.listdir(r"/data/govind/project/Background_without_leaves")
Blueberry__healthy = os.listdir(r"/data/govind/project/Blueberry__healthy")
Cherry__Powdery_mildew = os.listdir(r"/data/govind/project/Cherry__Powdery_mildew")
Cherry__healthy = os.listdir(r"/data/govind/project/Cherry__healthy")
Corn__Cercospora_leaf_spot_Gray_leaf_spot = os.listdir(r"/data/govind/project/Corn__Cercospora_leaf_spot Gray_leaf_spot")
Corn__Common_rust = os.listdir(r"/data/govind/project/Corn__Common_rust")
Corn__Northern_Leaf_Blight = os.listdir(r"/data/govind/project/Corn__Northern_Leaf_Blight")
Corn__healthy = os.listdir(r"/data/govind/project/Corn__healthy")
Grape__Black_rot = os.listdir(r"/data/govind/project/Grape__Black_rot")
Grape__Esca_ = os.listdir(r"/data/govind/project/Grape__Esca")
Grape__Leaf_blight = os.listdir(r"/data/govind/project/Grape__Leaf_blight")
Grape__healthy = os.listdir(r"/data/govind/project/Grape__healthy")
Orange__Haunglongbing = os.listdir(r"/data/govind/project/Orange__Haunglongbing")
Peach__Bacterial_spot = os.listdir(r"/data/govind/project/Peach__Bacterial_spot")
Peach__healthy = os.listdir(r"/data/govind/project/Peach__healthy")
Pepper__bell__Bacterial_spot = os.listdir(r"/data/govind/project/Pepper__bell__Bacterial_spot")
Pepper__bell_healthy = os.listdir(r"/data/govind/project/Pepper__bell___healthy")
Potato__Early_blight = os.listdir(r"/data/govind/project/Potato__Early_blight")
Potato__Late_blight = os.listdir(r"/data/govind/project/Potato__Late_blight")
Potato__healthy = os.listdir(r"/data/govind/project/Potato__healthy")
Raspberry__healthy = os.listdir(r"/data/govind/project/Raspberry__healthy")
Soybean__healthy = os.listdir(r"/data/govind/project/Soybean__healthy")
Squash__Powdery__mildew = os.listdir(r"/data/govind/project/Squash__Powdery_mildew")
Strawberry__Leaf_scorch = os.listdir(r"/data/govind/project/Strawberry__Leaf_scorch")
Strawberry__healthy = os.listdir(r"/data/govind/project/Strawberry__healthy")
Tomato__Bacterial_spot = os.listdir(r"/data/govind/project/Tomato__Bacterial_spot")
Tomato__Early_blight = os.listdir(r"/data/govind/project/Tomato__Early_blight")
Tomato__Late_blight = os.listdir(r"/data/govind/project/Tomato__Late_blight")
Tomato__Leaf_Mold = os.listdir(r"/data/govind/project/Tomato__Leaf_Mold")
Tomato__Septoria_leaf_spot = os.listdir(r"/data/govind/project/Tomato__Septoria_leaf_spot")
Tomato__Spider_mites_Two_spotted_spider_mite = os.listdir(r"/data/govind/project/Tomato__Spider_mites Two-spotted_spider_mite")
Tomato__Target_Spot = os.listdir(r"/data/govind/project/Tomato__Target_Spot")
Tomato__Tomato_Yellow_Leaf_Curl_Virus = os.listdir(r"/data/govind/project/Tomato__Tomato_Yellow_Leaf_Curl_Virus")
Tomato_Tomato_mosaic_virus = os.listdir(r"/data/govind/project/Tomato__Tomato_mosaic_virus")
Tomato_healthy =os.listdir(r"/data/govind/project/Tomato__healthy")

print("Number of Apple   Apple scab data : {}".format(len(Apple__Apple_scab)))
print("Number of Apple   Black rot data : {}".format(len(Apple__Black_rot)))
print("Number of Apple   Cedar apple rust data : {}".format(len(Apple__Cedar_apple_rust)))
print("Number of Apple   healthy data : {}".format(len(Apple__healthy)))
print("Number of Background without leaves data : {}".format(len(Background_without_leaves)))
print("Number of Blueberry   healthy data : {}".format(len(Blueberry__healthy)))
print("Number of Cherry   Powdery mildew data : {}".format(len(Cherry__Powdery_mildew)))
print("Number of Cherry   healthy data : {}".format(len(Cherry__healthy)))
print("Number of Corn   Cercospora leaf spot Gray leaf spot data : {}".format(len(Corn__Cercospora_leaf_spot_Gray_leaf_spot)))
print("Number of Corn   Common rust data : {}".format(len(Corn__Common_rust)))
print("Number of Corn   Northern Leaf Blight data : {}".format(len(Corn__Northern_Leaf_Blight)))
print("Number of Corn   healthy data : {}".format(len(Corn__healthy)))
print("Number of Grape   Black rot data : {}".format(len(Grape__Black_rot)))
print("Number of Grape__Esca_ data : {}".format(len(Grape__Esca_)))
print("Number of Grape   Leaf blight data : {}".format(len(Grape__Leaf_blight)))
print("Number of Grape   healthy data : {}".format(len(Grape__healthy)))
print("Number of Orange__Haunglongbing_ data : {}".format(len(Orange__Haunglongbing)))
print("Number of Peach__Bacterial_spot data : {}".format(len(Peach__Bacterial_spot)))
print("Number of Peach__healthy data : {}".format(len(Peach__healthy)))
print("Number of Pepper__bell__Bacterial_spot data : {}".format(len(Pepper__bell__Bacterial_spot)))
print("Number of Pepper__bell_healthy data : {}".format(len(Pepper__bell_healthy)))
print("Number of Potato__Early_blight data : {}".format(len(Potato__Early_blight)))
print("Number of Potato__Late_blight data : {}".format(len(Potato__Late_blight)))
print("Number of Potato__healthy data : {}".format(len(Potato__healthy)))
print("Number of Raspberry__healthy data : {}".format(len(Raspberry__healthy)))
print("Number of Soybean__healthy data : {}".format(len(Soybean__healthy)))
print("Number of Squash__Powdery__mildew data : {}".format(len(Squash__Powdery__mildew)))
print("Number of Strawberry__Leaf_scorch data : {}".format(len(Strawberry__Leaf_scorch)))
print("Number of Strawberry__healthy data : {}".format(len(Strawberry__healthy)))
print("Number of Tomato__Bacterial_spot data : {}".format(len(Tomato__Bacterial_spot)))
print("Number of Tomato__Early_blight data : {}".format(len(Tomato__Early_blight)))
print("Number of Tomato__Late_blight data : {}".format(len(Tomato__Late_blight)))
print("Number of Tomato__Leaf_Mold data : {}".format(len(Tomato__Leaf_Mold)))
print("Number of Tomato__Septoria_leaf_spot data : {}".format(len(Tomato__Septoria_leaf_spot)))
print("Number of Tomato__Spider_mites_Two_spotted_spider_mite data : {}".format(len(Tomato__Spider_mites_Two_spotted_spider_mite)))
print("Number of Tomato___Target_Spot data : {}".format(len(Tomato__Target_Spot)))
print("Number of Tomato_Tomato_mosaic_virus data : {}".format(len(Tomato_Tomato_mosaic_virus)))
print("Number of Tomato___Tomato_Yellow_Leaf_Curl_Virus data : {}".format(len(Tomato__Tomato_Yellow_Leaf_Curl_Virus)))
print("Number of Tomato_healthy data : {}".format(len(Tomato_healthy)))
target_size = (64, 64)

data = []

for i in range(len(Apple__Apple_scab)) :
    img = load_img(r"/data/govind/project/Apple__Apple_scab/{}".format(Apple__Apple_scab[i]))
    img = img.resize(target_size)
    arr = img_to_array(img)/255.0
    data.append([arr, 0])
for i in range(len(Apple__Black_rot)) :
    img = load_img(r"/data/govind/project/Apple__Black_rot/{}".format(Apple__Black_rot[i]))
    img = img.resize(target_size)
    arr = img_to_array(img)/255.0
    data.append([arr, 1])
for i in range(len(Apple__Cedar_apple_rust)) :
    img = load_img(r"/data/govind/project/Apple__Cedar_apple_rust/{}".format(Apple__Cedar_apple_rust[i]))
    img = img.resize(target_size)
    arr = img_to_array(img)/255.0
    data.append([arr, 2])
for i in range(len(Apple__healthy)) :
    img = load_img(r"/data/govind/project/Apple__healthy/{}".format(Apple__healthy[i]))
    img = img.resize(target_size)
    arr = img_to_array(img)/255.0
    data.append([arr, 3])
for i in range(len(Background_without_leaves)) :
    img = load_img(r"/data/govind/project/Background_without_leaves/{}".format(Background_without_leaves[i]))
    img = img.resize(target_size)
    arr = img_to_array(img)/255.0
    data.append([arr, 4])
for i in range(len(Blueberry__healthy)) :
    img = load_img(r"/data/govind/project/Blueberry__healthy/{}".format(Blueberry__healthy[i]))
    img = img.resize(target_size)
    arr = img_to_array(img)/255.0
    data.append([arr, 5])
for i in range(len(Cherry__Powdery_mildew)) :
    img = load_img(r"/data/govind/project/Cherry__Powdery_mildew/{}".format(Cherry__Powdery_mildew[i]))
    img = img.resize(target_size)
    arr = img_to_array(img)/255.0
    data.append([arr, 6])
for i in range(len(Cherry__healthy)) :
    img = load_img(r"/data/govind/project/Cherry__healthy/{}".format(Cherry__healthy[i]))
    img = img.resize(target_size)
    arr = img_to_array(img)/255.0
    data.append([arr, 7])
for i in range(len(Corn__Cercospora_leaf_spot_Gray_leaf_spot)) :
    img = load_img(r"/data/govind/project/Corn__Cercospora_leaf_spot Gray_leaf_spot/{}".format(Corn__Cercospora_leaf_spot_Gray_leaf_spot[i]))
    img = img.resize(target_size)
    arr = img_to_array(img)/255.0
    data.append([arr, 8])
for i in range(len(Corn__Common_rust)) :
    img = load_img(r"/data/govind/project/Corn__Common_rust/{}".format(Corn__Common_rust[i]))
    img = img.resize(target_size)
    arr = img_to_array(img)/255.0
    data.append([arr, 9])
for i in range(len(Corn__Northern_Leaf_Blight)) :
    img = load_img(r"/data/govind/project/Corn__Northern_Leaf_Blight/{}".format(Corn__Northern_Leaf_Blight[i]))
    img = img.resize(target_size)
    arr = img_to_array(img)/255.0
    data.append([arr, 10])
for i in range(len(Corn__healthy)) :
    img = load_img(r"/data/govind/project/Corn__healthy/{}".format(Corn__healthy[i]))
    img = img.resize(target_size)
    arr = img_to_array(img)/255.0
    data.append([arr, 11])
for i in range(len(Grape__Black_rot)) :
    img = load_img(r"/data/govind/project/Grape__Black_rot/{}".format(Grape__Black_rot[i]))
    img = img.resize(target_size)
    arr = img_to_array(img)/255.0
    data.append([arr, 12])
for i in range(len(Grape__Esca_)) :
    img = load_img(r"/data/govind/project/Grape__Esca/{}".format(Grape__Esca_[i]))
    img = img.resize(target_size)
    arr = img_to_array(img)/255.0
    data.append([arr, 13])
for i in range(len(Grape__Leaf_blight)) :
    img = load_img(r"/data/govind/project/Grape__Leaf_blight/{}".format(Grape__Leaf_blight[i]))
    img = img.resize(target_size)
    arr = img_to_array(img)/255.0
    data.append([arr, 14])
for i in range(len(Grape__healthy)) :
    img = load_img(r"/data/govind/project/Grape__healthy/{}".format(Grape__healthy[i]))
    img = img.resize(target_size)
    arr = img_to_array(img)/255.0
    data.append([arr, 15])
for i in range(len(Orange__Haunglongbing)) :
    img = load_img(r"/data/govind/project/Orange__Haunglongbing/{}".format(Orange__Haunglongbing[i]))
    img = img.resize(target_size)
    arr = img_to_array(img)/255.0
    data.append([arr, 16])
for i in range(len(Peach__Bacterial_spot)) :
    img = load_img(r"/data/govind/project/Peach__Bacterial_spot/{}".format(Peach__Bacterial_spot[i]))
    img = img.resize(target_size)
    arr = img_to_array(img)/255.0
    data.append([arr, 17])
for i in range(len(Peach__healthy)) :
    img = load_img(r"/data/govind/project/Peach__healthy/{}".format(Peach__healthy[i]))
    img = img.resize(target_size)
    arr = img_to_array(img)/255.0
    data.append([arr, 18])
for i in range(len(Pepper__bell__Bacterial_spot)) :
    img = load_img(r"/data/govind/project/Pepper__bell__Bacterial_spot/{}".format(Pepper__bell__Bacterial_spot[i]))
    img = img.resize(target_size)
    arr = img_to_array(img)/255.0
    data.append([arr, 19])

for i in range(len( Pepper__bell_healthy)) :
    img = load_img(r"/data/govind/project/Pepper__bell___healthy/{}".format(Pepper__bell_healthy[i]))
    img = img.resize(target_size)
    arr = img_to_array(img)/255.0
    data.append([arr, 20])
for i in range(len(Potato__Early_blight)) :
    img = load_img(r"/data/govind/project/Potato__Early_blight/{}".format(Potato__Early_blight[i]))
    img = img.resize(target_size)
    arr = img_to_array(img)/255.0
    data.append([arr, 21])
for i in range(len(Potato__Late_blight)) :
    img = load_img(r"/data/govind/project/Potato__Late_blight/{}".format(Potato__Late_blight[i]))
    img = img.resize(target_size)
    arr = img_to_array(img)/255.0
    data.append([arr, 22])
for i in range(len(Potato__healthy)) :
    img = load_img(r"/data/govind/project/Potato__healthy/{}".format(Potato__healthy[i]))
    img = img.resize(target_size)
    arr = img_to_array(img)/255.0
    data.append([arr, 23])  
for i in range(len(Raspberry__healthy)) :
    img = load_img(r"/data/govind/project/Raspberry__healthy/{}".format(Raspberry__healthy[i]))
    img = img.resize(target_size)
    arr = img_to_array(img)/255.0
    data.append([arr, 24])
for i in range(len(Soybean__healthy)) :
    img = load_img(r"/data/govind/project/Soybean__healthy/{}".format(Soybean__healthy[i]))
    img = img.resize(target_size)
    arr = img_to_array(img)/255.0
    data.append([arr, 25])
for i in range(len(Squash__Powdery__mildew)) :
    img = load_img(r"/data/govind/project/Squash__Powdery_mildew/{}".format(Squash__Powdery__mildew[i]))
    img = img.resize(target_size)
    arr = img_to_array(img)/255.0
    data.append([arr, 26])
for i in range(len(Strawberry__Leaf_scorch)) :
    img = load_img(r"/data/govind/project/Strawberry__Leaf_scorch/{}".format(Strawberry__Leaf_scorch[i]))
    img = img.resize(target_size)
    arr = img_to_array(img)/255.0
    data.append([arr, 27])

for i in range(len(Strawberry__healthy)) :
    img = load_img(r"/data/govind/project/Strawberry__healthy/{}".format(Strawberry__healthy[i]))
    img = img.resize(target_size)
    arr = img_to_array(img)/255.0
    data.append([arr, 28])
for i in range(len(Tomato__Bacterial_spot)) :
    img = load_img(r"/data/govind/project/Tomato__Bacterial_spot/{}".format(Tomato__Bacterial_spot[i]))
    img = img.resize(target_size)
    arr = img_to_array(img)/255.0
    data.append([arr, 29])
for i in range(len(Tomato__Early_blight)) :
    img = load_img(r"/data/govind/project/Tomato__Early_blight/{}".format(Tomato__Early_blight[i]))
    img = img.resize(target_size)
    arr = img_to_array(img)/255.0
    data.append([arr, 30])
for i in range(len(Tomato__Late_blight)) :
    img = load_img(r"/data/govind/project/Tomato__Late_blight/{}".format(Tomato__Late_blight[i]))
    img = img.resize(target_size)
    arr = img_to_array(img)/255.0
    data.append([arr, 31])

for i in range(len(Tomato__Leaf_Mold)) :
    img = load_img(r"/data/govind/project/Tomato__Leaf_Mold/{}".format(Tomato__Leaf_Mold[i]))
    img = img.resize(target_size)
    arr = img_to_array(img)/255.0
    data.append([arr, 32])
for i in range(len(Tomato__Septoria_leaf_spot)) :
    img = load_img(r"/data/govind/project/Tomato__Septoria_leaf_spot/{}".format(Tomato__Septoria_leaf_spot[i]))
    img = img.resize(target_size)
    arr = img_to_array(img)/255.0
    data.append([arr, 33])
for i in range(len(Tomato__Spider_mites_Two_spotted_spider_mite)) :
    img = load_img(r"/data/govind/project/Tomato__Spider_mites Two-spotted_spider_mite/{}".format(Tomato__Spider_mites_Two_spotted_spider_mite[i]))
    img = img.resize(target_size)
    arr = img_to_array(img)/255.0
    data.append([arr, 34])
for i in range(len(Tomato__Target_Spot)) :
    img = load_img(r"/data/govind/project/Tomato__Target_Spot/{}".format(Tomato__Target_Spot[i]))
    img = img.resize(target_size)
    arr = img_to_array(img)/255.0
    data.append([arr, 35])
for i in range(len(Tomato_Tomato_mosaic_virus)) :
    img = load_img(r"/data/govind/project/Tomato__Tomato_mosaic_virus/{}".format(Tomato_Tomato_mosaic_virus[i]))
    img = img.resize(target_size)
    arr = img_to_array(img)/255.0
    data.append([arr, 36])
for i in range(len(Tomato__Tomato_Yellow_Leaf_Curl_Virus)) :
    img = load_img(r"/data/govind/project/Tomato__Tomato_Yellow_Leaf_Curl_Virus/{}".format(Tomato__Tomato_Yellow_Leaf_Curl_Virus[i]))
    img = img.resize(target_size)
    arr = img_to_array(img)/255.0
    data.append([arr, 37])
for i in range(len(Tomato_healthy)) :
    img = load_img(r"/data/govind/project/Tomato__healthy/{}".format(Tomato_healthy[i]))
    img = img.resize(target_size)
    arr = img_to_array(img)/255.0
    data.append([arr, 38])
X, Y = [item for item in zip(*data)]
X = np.array(X)
Y = np.array(Y)
X.shape
X_train, X_test, Y_train , Y_test = train_test_split(X, Y, train_size = 0.75, random_state = 42)
import tensorflow as tf
from tensorflow.keras import layers, Model
# Visualize training history
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
# Create a new model CNN
model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(39, activation='softmax')  # 4 classes in this example
])
# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
epochs = 20# You can increase this for better performance
batch_size = 32
# Fit the model
history = model.fit(X_train, Y_train,epochs=epochs,batch_size=batch_size, validation_split=0.33)
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'],label="training accuracy")
plt.plot(history.history['val_accuracy'],label="testing accuracy")
plt.title('training and testing accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
loss,accuracy=model.evaluate(X_train, Y_train)
print(f'train Loss: {loss:.4f}')
print(f'train Accuracy: {accuracy:.4f}')
loss,accuracy = model.evaluate(X_test, Y_test)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')
model.save("/data/govind/project/leafmodel.h5")
model_CNN1= load_model(r"/data/govind/project/leafmodel.h5")
model_CNN1.custom_name = "CNN1"
models=[model_CNN1]
for model in models:
   
    if hasattr(model, 'custom_name'):
        print("Model Name:", model.custom_name)
    else:
        print("Model Name:", model.name)
       
    y_predict = model.predict(X_test)
    Y_predict = np.argmax(y_predict, axis=1)
   
    accuracy = accuracy_score(Y_test, Y_predict)
    print('Test Accuracy = %.4f' % accuracy)
   
    matrix = confusion_matrix(Y_test, Y_predict)
    tp = matrix[1, 1]
    fp = matrix[0, 1]
    tn = matrix[0, 0]
    fn = matrix[1, 0]
   
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    f1 = 2 * (precision * recall) / (precision + recall)
    kappa = (accuracy - (1 - accuracy)) / (1 - (1 - accuracy))
   
    print(matrix)
    print('True Positives:', tp)
    print('False Positives:', fp)
    print('True Negatives:', tn)
    print('False Negatives:', fn)
   
    print('Recall:', recall)
    print('Specificity:', specificity)
    print('Precision:', precision)
    print('F1 Score:', f1)
    print('Cohen\'s Kappa:', kappa)
   
    params = model.count_params()
    print("Parameters: {:.4f}".format(params))
   
    print("-" * 50)
import tkinter as tk
from tkinter import ttk,filedialog
import cv2
import PIL.Image, PIL.ImageTk
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model
class LeafDiseaseDetectorApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Leaf Disease Detector")

        self.model_path = r"/data/govind/project/leafmodel.h5"
        self.target_size = (64, 64)
        self.class_labels = {0: "Apple___Apple_scab",
                             1: "Apple___Black_rot",
                             2: "Apple___Cedar_apple_rust",
                             3: "Apple___healthy",
                             4: "Background_without_leaves",  
                             5: "Blueberry___healthy",
                             6: "Cherry___Powdery_mildew",
                             7: "Cherry___healthy",
                             8: "Corn___Cercospora_leaf_spot Gray_leaf_spot",
                             9: "Corn___Common_rust",
                             10: "Corn___Northern_Leaf_Blight",
                             11: "Corn___healthy",
                             12: "Grape___Black_rot",
                             13: "Grape___Esca_(Black_Measles)",
                             14: "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
                             15: "Grape__healthy",
                             16: "Orange___Haunglongbing_(Citrus_greening)",
                             17: "Peach___Bacterial_spot",
                             18: "Peach___healthy",
                             19: "Pepper,_bell___Bacterial_spot",
                             20: "Pepper,_bell___healthy",
                             21: "potato___early_blight",
                             22: "Potato___Late_blight",
                             23: "Potato___healthy",
                             24: "Raspberry___healthy",
                             25: "Soybean___healthy",
                             26: "Squash___Powdery_mildew",
                             27: "Strawberry___Leaf_scorch",
                             28: "Strawberry___healthy",
                             29: "Tomato___Bacterial_spot",
                             30: "Tomato___Early_blight",
                             31: "Tomato___Late_blight",
                             32: "Tomato___Leaf_Mold",
                             33: "Tomato___Septoria_leaf_spot",
                             34: "Tomato___Spider_mites Two-spotted_spider_mite",
                             35: "Tomato___Target_Spot",
                             36: "Tomato___Tomato_mosaic_virus",
                             37: "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
                             38: "Tomato___healthy"}
        self.model = load_model(self.model_path)
        self.label = ttk.Label(master)
        self.label.pack()

        self.result_label = ttk.Label(master, text="")
        self.result_label.pack()

        self.start_camera_button = ttk.Button(master, text="Start Camera", command=self.start_camera)
        self.start_camera_button.pack()

        self.upload_photo_button = ttk.Button(master, text="Upload Photo", command=self.upload_photo)
        self.upload_photo_button.pack()

        self.predict_button = ttk.Button(master, text="Predict", command=self.predict_disease)
        self.predict_button.pack()

        self.quit_button = ttk.Button(master, text="Quit", command=self.quit)
        self.quit_button.pack()

        self.camera = None
        self.current_image = None

    def start_camera(self):
        self.camera = cv2.VideoCapture(0)
        self.capture()

    def capture(self):
        if self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (640, 480))

                self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
                self.label.config(image=self.photo)
                self.label.image = self.photo

        if self.camera:
            self.master.after(10, self.capture)

    def upload_photo(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            image = cv2.imread(file_path)
            if image is not None:
                self.current_image = image
                self.display_image(image)
            else:
                print("Invalid image file")

    def display_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (640, 480))

        self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
        self.label.config(image=self.photo)
        self.label.image = self.photo

    def predict_disease(self):
        if self.current_image is not None:
            prediction_result = self.process_and_predict(self.current_image)
            self.result_label.config(text="Prediction Result: " + prediction_result)
        elif self.camera is not None and self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret:
                prediction_result = self.process_and_predict(frame)
                self.result_label.config(text="Prediction Result: " + prediction_result)
        else:
            print("No image or camera found.")

    def process_and_predict(self, image):
        # Preprocess the image
        image = cv2.resize(image, self.target_size)
        image = img_to_array(image) / 255.0
        image = np.expand_dims(image, axis=0)
       
        # Make predictions
        predictions = self.model.predict(image)
       
        # Interpret predictions
        predicted_class_index = np.argmax(predictions, axis=1)
        predicted_class_label = self.class_labels.get(predicted_class_index[0], "Unknown")
       
        return predicted_class_label

    def quit(self):
        if self.camera is not None:
            self.camera.release()
        self.master.destroy()


def main():
    root = tk.Tk()
    app = LeafDiseaseDetectorApp(root)
    root.mainloop()
if __name__ == "__main__":
    main()
