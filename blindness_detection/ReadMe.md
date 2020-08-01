# APTOS 2019 Blindness Detection
In the challenge the goal was to detect [diabetic retinopathy](https://nei.nih.gov/health/diabetic/retinopathy) and it's severity.
I tried to solve the challenge by training a model myself from scratch to learn more. It was also a challenge which I was personally
interested because my grandmother is blind. Not because of diabetes, but because of Cataracts, so I know how much a live can 
change when you lose your vision. Cataracts should also be able to be detected early and cost-efficient with the help of computer vision.

## Challenge
The Challenge can be found here on Kaggle: [APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection/overview)
The Data was made up of .pngs and .csvs. The name of the pictures corresponds to a row in a csv-file with the severity of the
diabetic retinopathy. Both the .pngs were split up in train and test. The test-files were used to create the final submission file, 
thus they weren't labeled.

## My approach
I didn't want to use a pretrained model just because I just learned about convolutional networks on 
[Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning) and I wanted to apply my new knowledge.

Generally, I went a standard-approach after seeing that applying filters (Edge Filters) to the images didn't really improve my score. With the help of Keras's ImageDataGenerator I created new Data by croping and zooming images. 
On each plateau I reduced the Learning Rate and stopped training the model when the [Cohen's kappa](https://en.wikipedia.org/wiki/Cohen%27s_kappa) wouldn't improve anymore. 

On a kaggle-dataset I also saw the idea to not predict classes, but to predict arrays, e.g. a Severity of "2" would be labeled as following: [0, 0, 1, 0, ,0].
Because a severity of "2" also contains the severity of "0", "1", "2", the labels get enhanced to [1, 1, 1, 0, 0]. 
This is also why the Accuracy is pretty high while the Cohen's Kappa Score is only "fair". 

This was run on [Euler](https://scicomp.ethz.ch/wiki/Euler).

## Results
Validation Accuracy: 0.8484 
Validation Cohen's Kappa Score: 0.4951

## Todo

* Use a pretrained model. I saw other participants using a pre-trained model with transfer-learning getting amazing scores (Cohen's Kappa Score of over 0.8).
* Increase the size of this NN and experiment more
* Clean the files, I think there are many packages which were used at the beginning (e.g. the filters) but not anymore.
* Evaluate the lost in Accuracy when using image preprocessing filters to reduce the file-size.
