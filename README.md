# CAPTCHA Solver

## Introduction
This project creates a couple of model architectures that given a CAPTCHA image, it predicts the three letters in that image. Note that the model architectures developed are trained and evaluated based on Google's old CAPTCHA system - where a user is asked to input characters seen given an image.

## Project motivation
This project has been created because it considers two key fields in deep learning: _computer vision_ and _natural language processing_. As a result, I want to consider two model architectures: An end-to-end convolutional neural network & a sequence-to-sequence network.

### Model architectures

#### 1) End-to-end covolutional neural network
#### A typical CAPTCHA CNN solution:
Through my research when solving this problem, people have approached it as follows:
1) Pre-processing the image (e.g., resizing, turning it into grayscale image, etc.),
2) Segment each character separately,
3) For every character, put it through a convolutional neural network and predict the closest character it resembles, and
4) Concatenate all predicted characters.

These steps seemed to handle CAPTCHA images that look like this:
(Add image here later)

Now what if the image contained overlapping characters like this:
(Add image here later)

#### 2) Sequence-to-sequence network
(Add things here later)

## Running this project
(Add things here later)

# Notes (format this later)
- CNN architecture for CAPTCHACNNClassifier: http://www.cs.sjsu.edu/faculty/pollett/masters/Semesters/Spring15/geetika/CS298%20Slides%20-%20PDF

