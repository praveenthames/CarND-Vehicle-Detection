##Writeup

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/CarAndNotCar.jpg
[image2]: ./output_images/HogFeaturesWithGray.jpg
[image3]: ./output_images/HogFeaturesWithHLS.jpg
[image4]: ./output_images/HogFeaturesWithYCrCb.jpg
[image5]: ./output_images/SpatialFeatures.jpg
[image6]: ./output_images/ColorHistogramFeatures.jpg
[image7]: ./output_images/RawAndNormalizedFeatures.jpg
[image8]: ./output_images/AllSizedWindows.jpg
[image9]: ./output_images/DrawBoxAllTestImages.jpg
[image10]: ./output_images/HeatmapAndLabelAllTestImages.jpg
[image11]: ./output_images/IntegratedHeatmapAndLabel.jpg
[image12]: ./output_images/BoundingBoxOnLastFrame.jpg
[video1]: ./project_video.mp4
[video2]: ./output_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Overview of the source code

####1. vehicle-detection.ipynb :  

This IPython notebook contains the logic for reading all the car and non car images (that were downloaded from the Udacity project on Github), extracting features (spatial, color histogram and hog), splitting the dataset into training set and test set, defining a Linear SVC classifier, sliding window search to identify potential feature matches for cars, adding heatmaps for the identified locations, applying threshold and finally reducing false positives.

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the code cells 3 through 7 of the IPython notebook vehicle-detection.ipynb.    

I started by reading in all the `vehicle` and `non-vehicle` images.  There is a total count of 8792  cars and 8968  non-cars of size:  (64, 64, 3). Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image4]


####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and here are some comparisons.

![alt text][image2]

![alt text][image3]

Comparing the images above, I observed that on `orientations=8`, `pixels_per_cell=(6, 6)` and `cells_per_block=(2, 2)` on the gray scale, I get a pretty good feature set and hence settled down on those values for the parameters to extract hog features.

In addition to the hog features, I also extracted the Spatial and Color histogram features.

The code for Spatial feature is contained in code cells 4 and 11. I reduced the size of the image to (8,8) so that the number of features generated can be managed and used ravel to compute the spatial feature vector. We use raw pixel intensities of the image in RGB and HLS color spaces as features. The car object in the image in the S channel of HLS appears to be more clearer than the other channels in the color space. Below is an example of Spatial feature map of a car image in the S channel of HLS color space. 

![alt text][image5]

The code for Color histogram is contained in the code cells 5 and 12. We use Color Histogram to extract features so that we can detect the different appearances of the car. The histogram is computed for each channel in the RGB and HLS color spaces and then binning is performed on the histograms. The Color Histogram remove the structural relation and allow more flexibility to the variance of the image. I started with `hist_bins=12` and upon increasing or decreasing this value, I did not see a significant change in the features and hence settled down with the same value for this parameter. The following is an example of an image from training data set and its color histogram in RGB and HLS color spaces.

![alt text][image6]

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code for extracting the car and not car features and normalizing the feature set is in the code cell 16.
The code for the SVC classifier is in the cell 17.  
The code for the RandomForestClassifier are in the cells 18 through 20.

All the images in the vehicles and non-vehicles dataset are all in png format which will be in the scale of 0-1. However, the images under the test_images and the image frames extracted from the project_video are in jpg format, which will be in a scale of 0-255. So as a first step, I make sure that before I train a classifier, I convert the car and not car images into scale of 0-255. 

The car and not car features are extracted, concatenated and then are normalized using `StandardScalar` from `sklearn.preprocessing`. Below is the image showing the raw and normalized features.

![alt text][image7]

Since we just concatenated the car and non-car dataset, all the car data will be on one end and the non-car will be on the other end. So the dataset is shuffled using `shuffle` from `sklearn.utils` and then the data is split into training and test set(though this is not our test, but our validation set, we still call it test set). 
 
I trained using the LinearSVC classifier first, even though I was getting an accuracy of ~96%, it did not detect some cars which were very obvious. Based on some slack channel references and discussions, I explored the `RandomForestClassifier` from `sklearn.ensemble` and that was giving an accuracy of ~97% and was able to do a better job at detecting the cars in the video. The RandomForestClassifier has the following parameters - `n_estimators`, `max_features`, `max_depth` and `min_samples_leaf`. The code cell 18 is where I tune these parameters. The metric `auroc` from `sklearn.metrics` improves and stabilizes at `n_estimators=100`, `max_features=2`, `max_depth=25` and `min_samples_leaf=2`. 


###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code cell 23 is the function that slides the window of a given size over the image. This function is unmodified from the version on the lecture and quiz. This function returns the list of windows that are potential matches for car features.

The code cell 24 and 25 is where we set the limits of the image in x and y axis and start identifying the different sizes of windows possible within that range. For the limits of y, we set between 400 to 700, because we are only interested in the area of the image from the horizon and below until the front of our car. For the limits of x, we are interested in values above 500, that is where the left yellow line is. For the min and max window sizes we picked (80,80) and (160,160) because the baseline I chose was (96,96) as discussed in the project Q&A and then searched for 4 different possible window sizes with an overlap of 0.5. After getting the different window sizes and x_start_stop and y_start_stop, we run the `slide_window` on a sample image from the test_images and here is the output.


![alt text][image8]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on four scales using gray scale HOG features plus spatially binned color and histograms of color in the feature vector and classified using a RandomForestClassifier, which provided a nice result. RandomForestClassifier has a function called `predict_proba` that predicts the class probabilities of the features. The predicted class probabilities of an input sample are computed as the mean predicted class probabilities of the trees in the forest. The class probability of a single tree is the fraction of samples of the same class in a leaf. Refer http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier.predict_proba

To optimize the performance of the classifier, I played around with setting a threshold for the prediction accuracy. I settled at a threshold of 0.51 since anything above that was missing to detect some real cars. Here are some example images:

![alt text][image9]

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to the final output video](./output_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  


Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image10]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image11]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image12]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I defined a new class called Car that contains the attributes - average_centroid, width, height and detected which is the moving average of the car. We use this class to hold cars detected in each frame which becomes easy to compare the two consecutive frames for the distance of the detected cars.  
I used the [Moving Mean algorithm](https://en.wikipedia.org/wiki/Moving_average) to recalculate the heat map in every frame. I include 90% of the last frame heat map and include 10% of the heatmap of the new frame. This way the heatmap is smoothened. I also used an upper and lower threshold values for the heat map. The same Moving Average algorithm is used to recompute the , already detected car's average centroid, width and height in every frame. One difficulty I faced was with fine tuning the parameters for extracting the features - It took more time to manually try different combinations of them. 
 
If I pursue the project further, I would like to explore even more classifiers which might yield even better results. I would also try to integrate the advanced lane finding logic in here to so as to filter out the false positives which fall outside a lane line. Currently the pipeline might not work when the video is recorded from a car driving on the right most lane. Another situation we have not tested is the two-way roads with traffic coming against the camera when its dark. Another scenario that I would like to test is when there are traffic signs and bill boards containing images like that of a car.

References and Inspirations:
Project 5 Udacity Live Video Session : https://www.youtube.com/watch?v=P2zwrTM8ueA&feature=youtu.be
Dalaska : https://github.com/Dalaska/CarND-P5-Vehicle-Detection-and-Tracking
Slack channels and discussion forums
