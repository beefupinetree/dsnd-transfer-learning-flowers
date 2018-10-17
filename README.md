
README
# **Transfer Learning for Image Classification**
<p align="center"><img src="/img/cover.png?raw=true"/></p>



---

## Overview

Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications. 

In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories, you can see a few examples below. 


The project is broken down into multiple steps:

* Load and preprocess the image dataset
* Train the image classifier on your dataset
* Use the trained classifier to predict image content

We'll lead you through each part which you'll implement in Python.

When you've completed this project, you'll have an application that can be trained on any set of labeled images. Here your network will be learning about flowers and end up as a command line application. But, what you do with your new skills depends on your imagination and effort in building a dataset. For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it. Go build your own dataset and make something new.

First up is importing the packages you'll need. It's good practice to keep all the imports at the beginning of your code. As you work through this notebook and find you need to import a package, make sure to add the import up here.

```python
import json
import time
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from collections import OrderedDict
from PIL import Image
import numpy as np
import os
import helper
```

## Load the data

Here you'll use `torchvision` to load the data ([documentation](http://pytorch.org/docs/master/torchvision/transforms.html#)). The data should be included alongside this notebook, otherwise you can [download it here](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz). The dataset is split into three parts, training, validation, and testing. For the training, you'll want to apply transformations such as random scaling, cropping, and flipping. This will help the network generalize leading to better performance. You'll also need to make sure the input data is resized to 224x224 pixels as required by the pre-trained networks.

The validation and testing sets are used to measure the model's performance on data it hasn't seen yet. For this you don't want any scaling or rotation transformations, but you'll need to resize then crop the images to the appropriate size.

For all three sets you'll need to normalize the means and standard deviations of the images to what the network expects. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`. This converts the values of each color channel to be between -1 and 1 instead of 0 and 1.


```python
data_dir = 'flowers'
train_dir = os.path.join(data_dir, "train")
valid_dir = os.path.join(data_dir, "valid")
test_dir = os.path.join(data_dir, "test")
```


```python
# Define your transforms for the training, validation, and testing sets
data_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.RandomRotation(45),
                                      transforms.RandomHorizontalFlip(p=0.4),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                              (0.485, 0.456, 0.406),
                                              (0.229, 0.224, 0.225))])

data_transforms_test = transforms.Compose([transforms.Resize(224),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize(
                                                   (0.485, 0.456, 0.406),
                                                   (0.229, 0.224, 0.225))])

# Load the datasets with ImageFolder
image_datasets_train = datasets.ImageFolder(train_dir,
                                            transform=data_transforms)
image_datasets_valid = datasets.ImageFolder(valid_dir,
                                            transform=data_transforms)
image_datasets_test = datasets.ImageFolder(test_dir,
                                           transform=data_transforms_test)

# Using the image datasets and the trainforms, define the dataloaders
trainloader = DataLoader(image_datasets_train, batch_size=64, shuffle=True)
validloader = DataLoader(image_datasets_valid, batch_size=64, shuffle=True)
testloader = DataLoader(image_datasets_test, batch_size=32, shuffle=False)
```

## Sample images

```python
# Display some pictures from the testing set
data_iter = iter(testloader)
images, labels = next(data_iter)
fig, axes = plt.subplots(figsize=(10, 4), ncols=4)
for ii in range(4):
    ax = axes[ii]
    helper.imshow(images[ii], ax=ax)
```
![Sample images](./img/sample.png)

### Label mapping

You'll also need to load in a mapping from category label to category name. You can find this in the file `cat_to_name.json`. It's a JSON object which you can read in with the [`json` module](https://docs.python.org/2/library/json.html). This will give you a dictionary mapping the integer encoded categories to the actual names of the flowers.


```python
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
```

# Building and training the classifier

Now that the data is ready, it's time to build and train the classifier. As usual, you should use one of the pretrained models from `torchvision.models` to get the image features. Build and train a new feed-forward classifier using those features.

We're going to leave this part up to you. If you want to talk through it with someone, chat with your fellow students! You can also ask questions on the forums or join the instructors in office hours.

Refer to [the rubric](https://review.udacity.com/#!/rubrics/1663/view) for guidance on successfully completing this section. Things you'll need to do:

* Load a [pre-trained network](http://pytorch.org/docs/master/torchvision/models.html) (If you need a starting point, the VGG networks work great and are straightforward to use)
* Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
* Train the classifier layers using backpropagation using the pre-trained network to get the features
* Track the loss and accuracy on the validation set to determine the best hyperparameters

We've left a cell open for you below, but use as many as you need. Our advice is to break the problem up into smaller parts you can run separately. Check that each part is doing what you expect, then move on to the next. You'll likely find that as you work through each part, you'll need to go back and modify your previous code. This is totally normal!

When training make sure you're updating only the weights of the feed-forward network. You should be able to get the validation accuracy above 70% if you build everything right. Make sure to try different hyperparameters (learning rate, units in the classifier, epochs, etc) to find the best model. Save those hyperparameters to use as default values in the next part of the project.


```python
# Choose the pretrained network to transfer
model = models.vgg19(pretrained=True)

# Freeze parameters in pretrained network so we don't waste time by backpropagating through them
for param in model.parameters():
    param.requires_grad = False

# Design the layers and neurons of your classifier network
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 10000)),
                          ('relu', nn.ReLU()),
                          ('d1', nn.Dropout(p=0.2)),
                          ('fc2', nn.Linear(10000, len(cat_to_name))),
                          ('d2', nn.Dropout(p=0.2)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

model.classifier = classifier

# Choose GPU for training, if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Select the criterion for calculating the loss function
criterion = nn.NLLLoss()

# Choose the algorithm to change the weights during backpropagation
# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
```

```python
# model = train.train_model(image_datasets, arch='alexnet', gpu=True, epochs=14, hidden_units=4096)
model.to(device)

epochs = 10
steps = 0
print_every = 40
for e in range(epochs):
    running_loss = 0
    tot_time = time.time()
    model.train()
    for inputs, labels in iter(trainloader):
        steps += 1
        # Move input and label tensors to the GPU
        inputs, labels = inputs.to(device), labels.to(device)
        start = time.time()
        optimizer.zero_grad()

        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if steps % print_every == 0:
            # Make sure network is in eval mode for inference
            model.eval()

            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                test_loss, accuracy = validation(model, validloader, criterion)
            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                  "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
                  "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))

            running_loss = 0

            # Make sure training is back on
            model.train()
            running_loss = 0
    print("Time per epoch = {:.3f}".format(time.time()-tot_time), "seconds.")
```

# Insert gif of training


## Testing your network

It's good practice to test your trained network on test data, images the network has never seen either in training or validation. This will give you a good estimate for the model's performance on completely new images. Run the test images through the network and measure the accuracy, the same way you did validation. You should be able to reach around 70% accuracy on the test set if the model has been trained well.


```python
correct = 0
total = 0
model.to(device)
model.eval()
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the', len(testloader), 'test images: %d %%'
      % (100 * correct / total))

```

    Using GPU
    Accuracy of the network on the test images: 67 %


## Save the checkpoint

Now that your network is trained, save the model so you can load it later for making predictions. You probably want to save other things such as the mapping of classes to indices which you get from one of the image datasets: `image_datasets['train'].class_to_idx`. You can attach this to the model as an attribute which makes inference easier later on.

```model.class_to_idx = image_datasets['train'].class_to_idx```

Remember that you'll want to completely rebuild the model later so you can use it for inference. Make sure to include any information you need in the checkpoint. If you want to load the model and keep training, you'll want to save the number of epochs as well as the optimizer state, `optimizer.state_dict`. You'll likely want to use this trained model in the next part of the project, so best to save it now.


```python
# Save a checkpoint 
model.class_to_idx = image_datasets_train.class_to_idx

checkpoint = {
              'state_dict': model.state_dict(),
              'image_datasets': model.class_to_idx,
              'epochs': epochs,
              'optimizer': optimizer.state_dict(),
              'classifier': model.classifier,
              'model': "vgg19",
             }

torch.save(checkpoint, 'checkpoint.pth')
```

## Loading the checkpoint

At this point it's good to write a function that can load a checkpoint and rebuild the model. That way you can come back to this project and keep working on it without having to retrain the network.


```python
def load_checkpoint(filepath):
        checkpoint = torch.load(filepath)
        model = getattr(models, checkpoint['model'])
        model = model(pretrained=True)
        model.classifier = checkpoint['classifier']
        model.load_state_dict(checkpoint['state_dict'])
        model.class_to_idx = checkpoint['image_datasets']
        for param in model.parameters():
            param.requires_grad = False
        return model


model = load_checkpoint('checkpoint.pth')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_dir = 'flowers'
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
```

# Inference for classification

Now you'll write a function to use a trained network for inference. That is, you'll pass an image into the network and predict the class of the flower in the image. Write a function called `predict` that takes an image and a model, then returns the top $K$ most likely classes along with the probabilities. It should look like 

First you'll need to handle processing the input image such that it can be used in your network. 

## Image Preprocessing

You'll want to use `PIL` to load the image ([documentation](https://pillow.readthedocs.io/en/latest/reference/Image.html)). It's best to write a function that preprocesses the image so it can be used as input for the model. This function should process the images in the same manner used for training. 

First, resize the images where the shortest side is 256 pixels, keeping the aspect ratio. This can be done with the [`thumbnail`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) or [`resize`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) methods. Then you'll need to crop out the center 224x224 portion of the image.

Color channels of images are typically encoded as integers 0-255, but the model expected floats 0-1. You'll need to convert the values. It's easiest with a Numpy array, which you can get from a PIL image like so `np_image = np.array(pil_image)`.

As before, the network expects the images to be normalized in a specific way. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`. You'll want to subtract the means from each color channel, then divide by the standard deviation. 

And finally, PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array. You can reorder dimensions using [`ndarray.transpose`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.transpose.html). The color channel needs to be first and retain the order of the other two dimensions.


```python
# Process a PIL image for use in a PyTorch model
ddef process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    # Resizing to 256 pixels while keeping aspect ratio
    min_length = 256
    img = Image.open(image)
    width, height = img.size    # Get dimensions
    if width < height:
        img.thumbnail((256, height))
    else:
        img.thumbnail((width, 256))

    # Center crop
    if min(width, height) < 224:    # Skips cropping pictures with less than
                                    # 224 pixels in either dimension
        return
    w, h = 224, 224     # New dimensions for center crop
    width, height = img.size    # Get new dimensions after resizing
    left = (width - w) // 2
    top = (height - h) // 2
    right = (width + w) // 2
    bottom = (height + h) // 2
    img = img.crop((left, top, right, bottom))
    # normalizing values of color channels
    # converting to tensor to normalize
    to_tensor = transforms.ToTensor()
    img = to_tensor(img)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean, std)
    img = normalize(img)
    img = np.array(img)
    img_trans = img.transpose()
    return img_trans
```

To check your work, the function below converts a PyTorch tensor and displays it in the notebook. If your `process_image` function works, running the output through this function should return the original image (except for the cropped out portions).


```python
def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    # image = image.transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when
    # displayed
    image = np.clip(image, 0, 1)
    plt.title(title)
    ax.imshow(image)

    return ax
```


```python
# %matplotlib inline

img = (data_dir + '/test/1//image_06752.jpg')

# Test function on a single picture
imshow(process_image(img))
```


![Sample image output](./img/output.png)


## Class Prediction

Once you can get images in the correct format, it's time to write a function for making predictions with your model. A common practice is to predict the top 5 or so (usually called top-$K$) most probable classes. You'll want to calculate the class probabilities then find the $K$ largest values.,

To get the top $K$ largest values in a tensor use [`x.topk(k)`](http://pytorch.org/docs/master/torch.html#torch.topk). This method returns both the highest `k` probabilities and the indices of those probabilities corresponding to the classes. You need to convert from these indices to the actual class labels using `class_to_idx` which hopefully you added to the model or from an `ImageFolder` you used to load the data ([see here](#Save-the-checkpoint)). Make sure to invert the dictionary so you get a mapping from index to class as well.

Again, this method should take a path to an image and a model checkpoint, then return the probabilities and classes.

```python
img = (data_dir + '/test/1/image_06752.jpg')
probs, classes = predict(img, model)
print(probs)
print(classes)
> [0.907157   0.04348968 0.01251764 0.01162242 0.00867847]
> ['19', '1', '93', '76', '86']
```

## Sanity Checking

Now that you can use a trained model for predictions, check to make sure it makes sense. Even if the testing accuracy is high, it's always good to check that there aren't obvious bugs. Use `matplotlib` to plot the probabilities for the top 5 classes as a bar graph, along with the input image. It should look like this:

<p align="center"><img src="/img/sanity_check.png?raw=true"/></p>

You can convert from the class integer encoding to actual flower names with the `cat_to_name.json` file (should have been loaded earlier in the notebook). To show a PyTorch tensor as an image, use the `imshow` function defined above.


```python
# TODO: Display an image along with the top 5 classes

img_path = (data_dir + '/test/18/image_04254.jpg')
img = process_image(img_path)

# function to handle a different path structure when labeling the input image


def check_structure(path): 
    try:
        ans = cat_to_name[path.split("/")[8]]
    except IndexError:
        return 'Anonymous'
    return ans


ttl = check_structure(img_path)
imshow(img, title=ttl)
probs, classes = predict(img_path, model)
names = []
for k in classes:
    names.append(cat_to_name[k])

# Fixing random state for reproducibility
np.random.seed(42)

plt.rcdefaults()
fig, ax = plt.subplots()

y_pos = np.arange(len(names))


ax.barh(y_pos, probs, align='center',
        color='green', ecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(names)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Probabilities')
ax.set_title('How confident is this model, really?')
plt.xlim([0.0, 1.0])   # set the range of the x-axis between 0 & 1
plt.show()
```
<p align="center"><img src="/img/result.png?raw=true"/></p>
<p align="center"><img src="/img/result_2.png?raw=true"/></p>
