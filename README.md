Faster RCNN Model Architecture

The Faster R-CNN model is an object detection network which integrates proposal generation step directly with the neural networks unlike its predecessors. Its main components include:
Backbone (ResNet):
A ResNet convolutional neural network is used to extract hierarchical feature maps from the input image.


Region Proposal Network (RPN):
The RPN slides over the feature maps and proposes candidate regions (which are known as anchors) that may contain objects.
Each anchor is scored with an objectness score and refined with bounding box regressors.


Region of Interest (RoI) Pooling:
The proposed regions are cropped and resized into a fixed size, enabling uniform input for the detection head.


Detection Head:
A fully connected classification head predicts the class label (pedestrian vs background).
A regression head refines the bounding box coordinates.


Pretrained Initialization:
The network is initialized with COCO pretrained weights, making sure that good feature representations help in pedestrian detection.
Only the final layers are adapted to the pedestrian detection task.

