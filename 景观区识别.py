import pandas as pd
import requests
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.preprocessing import OneHotEncoder
import glob
from skimage import io, transform
# 存放kaggle 数据集Google landmark数据集中的train.csv
content = pd.read_csv('train.csv')
# 设置可查看所有列内容
pd.set_option('display.max_columns', None)
# 选择景区landmark_id前15个景区进行训练
arr = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']
content_selected = content[content['landmark_id'].isin(arr)]
content_selected = content_selected.reset_index()
num = content_selected.shape[0]
# 从train.csv中的url下载相应图片，需要翻墙
def download_img(num):
    for i in range(num):
        img_url = content_selected.at[i, 'url']
        img_name = str(i) + '.jpg'
        folder = content_selected.at[i, 'landmark_id']
        path = 'img/' + folder + '/'
        isExists = os.path.exists(path)
        if not isExists:
            os.makedirs(path)
        try:
            img = requests.get(img_url)
            with open(path + img_name, 'ab') as f:
                f.write(img.content)
        except:
            pass
content_selected.to_csv('data.csv')
download_img(num)


def random_color(image, color_order=0):
    if color_order == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    elif color_order == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    return tf.clip_by_value(image, 0.0, 1.0)


def preprocess_for_train(image, height, width, bbox):
    if bbox is None:
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(tf.shape(image), bounding_boxes=bbox)
    random_image = tf.slice(image, bbox_begin, bbox_size)
    random_image = tf.image.resize_images(random_image, [height, width], method=np.random.randint(4))
    random_image = tf.image.random_flip_left_right(random_image)
    random_image = random_color(random_image, np.random.randint(2))
    return random_image


image_raw_data = tf.gfile.FastGFile(r'C:\Users\开元\Desktop\tensorflow_demo\img\0\0.jpg', 'rb').read()
with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)
    boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]])
    for i in range(1):
        print('hahaha')
        result = preprocess_for_train(img_data,600,600,boxes)
        plt.imshow(result.eval())
        plt.show()


w = 128
h = 128
c = 3
path = 'img/'


def One_Hot_Label_Encoding(labels):
    label = labels
    enc = OneHotEncoder()
    enc.fit(label)
    one_hot_labels = enc.transform(label).toarray()
    return one_hot_labels


def read_img(path):
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    print(cate)
    imgs = []
    labels = []
    for idx, folder in enumerate(cate):
        for im in glob.glob(folder + '/*.jpg'):
            img = io.imread(im)
            img = transform.resize(img, (w, h))
            imgs.append(img)
            label = int(folder.split('/')[-1])
            labels.append(label)
    labels = np.array(labels)
    newlabels = labels[:, np.newaxis]
    one_hot_labels = One_Hot_Label_Encoding(newlabels)
    return np.asarray(imgs, np.float32), one_hot_labels


data, labels = read_img(path)
print(data.shape)
print(labels.shape)
data_normalize = data / 255.0
num_example = data_normalize.shape[0]
arr = np.arange(num_example)
np.random.shuffle(arr)
data_normalize = data_normalize[arr]
labels = labels[arr]
ratio = 0.8
s = np.int(num_example * ratio)
x_train = data_normalize[:s]
y_train = labels[:s]
x_test = data_normalize[s:]
y_test = labels[s:]
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(9, 9),
                 input_shape=(w, h, c), activation='relu', padding='same'))
model.add(Dropout(rate=0.25))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Conv2D(filters=64, kernel_size=(9, 9),
                 activation='relu', padding='same'))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Flatten())
model.add(Dropout(rate=0.25))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(rate=0.25))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
model.summary()
train_history = model.fit(x_train, y_train,
            validation_split=0.2, epochs=20, batch_size=8, verbose=2)
scores = model.evaluate(x_test,y_test,verbose = 0)