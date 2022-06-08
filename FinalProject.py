import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os
import cv2 as cv
import numpy as np

def compute_confusion_matrix(true, pred):
    #K = len(np.unique(true)) # Number of classes
    K = 4
    result = np.zeros((K, K))
    for j in range(len(true)):
        result[true[j]][pred[j]] += 1
    return result


path = 'dataset2'

train_list = []
train_label = []
test_list = []
test_label = []
img_dim = 100
count = 0
label = []
for filename in os.listdir(path):
    file = os.path.join(path, filename)
    # Assign value as label for classification
    if "cloudy" in file:
        label = 0
    if "rain" in file:
        label = 1
    if "shine" in file:
        label = 2
    if "sunrise" in file:
        label = 3
    img = cv.imread(file)
    # Error handling for images not being read correctly
    if img is None:
        print("Error loading image: " + filename)
        continue


    # Resize images
    img = cv.resize(img, (img_dim, img_dim))
    # Convert from BRG to RGB
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # Normalize pixels to between 0 and 1
    img = img / 255.0
    # Add every fifth image to test set and all others to train set
    if (count % 5) == 0:
        test_list.append(img)
        test_label.append(np.array(label))
    else:
        train_list.append(img)
        train_label.append(np.array(label))
    count = count + 1

# One-hot encode the labels and convert training lists to np arrays
train_label = tf.one_hot(train_label, 4)
train_list = np.array(train_list)
test_label_not_oh = np.array(test_label)
test_label = tf.one_hot(test_label, 4)
test_list = np.array(test_list)

# Plot 10 photos
fig1 = plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.xticks([1])
    plt.yticks([1])
    plt.grid(False)
    plt.imshow(train_list[i * 80])

plt.show()

# Create model and add convolution and pooling layers
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_dim, img_dim, 3)))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Flatten model and apply dense layers and softmax output activation function
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(4, activation='softmax'))

model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_list, train_label, epochs=10,
                    validation_data=(test_list, test_label))

# Plot of epochs vs. model accuracy
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(test_list,  test_label, verbose=2)
print(test_acc)

path = "personal_images"

own_test_list = []
for filename in os.listdir(path):
    file = os.path.join(path, filename)

    img = cv.imread(file)
    if img is None:
        print("Error loading image: " + filename)
        continue
    # Resize images
    img = cv.resize(img, (img_dim, img_dim))
    # Convert from BRG to RGB
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # Normalize pixels to between 0 and 1
    img = img / 255.0

    own_test_list.append(img)

own_test_labels = [0, 0, 3, 0, 2,
                   0, 0, 2, 0, 0]

own_test_labels_oh = tf.one_hot(own_test_labels, 4)
own_test_list = np.array(own_test_list)

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.xticks([1])
    plt.yticks([1])
    plt.grid(False)
    plt.imshow(own_test_list[i])

plt.show()

# Compute confusion matrix of dataset
own_pred_dataset = model.predict(test_list)
own_pred_dataset_t = []
for i in range(len(own_pred_dataset)):
    own_pred_dataset_t.append(np.argmax(own_pred_dataset[i]))

confusion_mx_dataset = compute_confusion_matrix(test_label_not_oh, own_pred_dataset_t)
print("Dataset Confusion Matrix:")
print(confusion_mx_dataset)

correct_preds_dataset = 0
for i in range(len(confusion_mx_dataset)):
    correct_preds_dataset += confusion_mx_dataset[i][i]

print("True Positives: " + str(correct_preds_dataset))
print("True Positive Rate: " + str((correct_preds_dataset / len(test_list)) * 100) + "%")
print("False Positives: " + str((len(test_list) - correct_preds_dataset)))
print("False Positive Rate: " + str(((len(test_list) - correct_preds_dataset) / len(test_list) * 100)) + "%")


test_loss, test_acc = model.evaluate(own_test_list,  own_test_labels_oh, verbose=2)
print(test_acc)

# Compute confusion matrix of my own test images
own_pred = model.predict(own_test_list)
own_pred_t = []
for i in range(len(own_pred)):
    own_pred_t.append(np.argmax(own_pred[i]))

confusion_mx = compute_confusion_matrix(own_test_labels, own_pred_t)
print("Own Images Confusion Matrix:")
print(confusion_mx)
print("Actual labels:")
print(own_test_labels)
print("Predicted labels:")
print(own_pred_t)

correct_preds = 0
for i in range(len(confusion_mx)):
    correct_preds += confusion_mx[i][i]

print("True Positives: " + str(correct_preds))
print("True Positive Rate: " + str((correct_preds * 10)) + "%")
print("False Positives: " + str((10 - correct_preds)))
print("False Positive Rate: " + str(((10 - correct_preds) * 10)) + "%")

