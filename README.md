#RNN CHEF
## Generating Sequential Recipe Instructions using Food Images

## Team Members
1. Anastasiya Lazareva [@alazareva](https://github.com/alazareva) 
2. Paul Galloway [@paul-o-alto](https://github.com/paul-o-alto)
3. Sharath Rao [@sharathrao13](https://github.com/sharathrao13)


Basic Idea
* We scraped the images and recipes to form our own dataset.
* We extracted features from the image using Inception v3 (tensorflow model)
* We performed principal component analysis in order to reduce the dimension of the image features.
* The data is trained using an LSTM trained using Adam Optimizer and Cross entropy loss.
* Some interesting results are shown in the website.

Reference
We have modified and used the LSTM developed in Show Attend and Tell. The original code is available here -> https://github.com/jazzsaxmafia/show_attend_and_tell.tensorflow


