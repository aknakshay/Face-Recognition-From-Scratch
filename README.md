# Face-Recognition-From-Scratch
Implementing a Face Recognition CV Model using Online Triplet Mining on IMFDB Dataset


FaceNet is a deep learning model which learns mappings from face images to a compact Euclidian space and the distance between two embeddings correspond to the measure of similarity between faces. The embeddings can be used for face verification, clustering faces, etc.

The model uses triplet loss as the loss function.

For image embeddings, triplet loss is great way to create embeddings. In triplet loss, we take an image as an anchor, another image which is of the same person/object as positive example and another image which is not of the same person/object as negative example.
We try to create embeddings in a way that the distance between the embeddings of the anchor and positive image should be lower and the distance between the embeddings of the anchor and the negative image should be higher.

Triplets are generated online by selecting hard positive/negative examples from within a minibatch. The triplet images are transformed into 128 Dimensional embeddings and the distance between embeddings of positive and anchor is decreased and distance between embeddings of negative and anchor image is increased.

Steps
1.	Load Image Dataset
2.	Pre-process Dataset
a.	Detect Faces in the image using MTCNN and crop the image accordingly
b.	Affine transformations are applied to straighten the face
c.	Normalize the image
d.	Resize the image to the input size of 224 × 224
3.	Training (with Online Triplet Selection)
For minibatch in data:
a.	Train the network forward and store the 128-D embedding
b.	Calculate the distance between each of the image in the mini batch
c.	From an anchor, find semi hard and hard positives and negatives amongst the minibatch for each identity with alpha (margin) set to 2
d.	Using argmax of positive and argmin of negative, get the triplet to be used to calculate loss
e.	Back Propagate the feedback w.r.t loss
4.	Validation: Classification using SVM over embeddings
a.	Pre-Process the validation/test set
b.	Get 128-D embeddings using the facenet model
c.	Divide the dataset into train and test
d.	Train an SVM model using train set
e.	Test the model using test set

