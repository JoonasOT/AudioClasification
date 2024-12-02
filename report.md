# Project Work COMP.SGN.120

## Introduction
In this project work, the goal is to implement a binary audio classification model which is
able to classify a short audio sample to be from either a car or a tram. The algorithm is
implemented as a python script using common python audio signal processing libraries such as
librosa and scipy.


## Data description
The data used for training and testing the model is obtained from the
course's Freesound repository and it is assumed to be of adequate quality.
As the model is a binary classifier, there are only two classes, tram and car.
Most of the data has been collected by recording the audio of a car or
a tram passing by with a smartphone.

## Feature extraction
Three features were chosen to extract from the audio signals. MFCCs were obtained
because it is based on the Mel-scale which emphasizes the lower-frequency components
which are the frequencies of interest.
Spectral centroid is calculated to find out the center of mass of the frequency spectrum.
By listening to some of the data, the tram seems to have distinct higher frequency
components compared to the car, which hypothetically would be detected with the spectral
centroid feature.
Third feature is...