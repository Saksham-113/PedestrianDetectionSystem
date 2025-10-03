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
Dataset Preparation
The dataset consisted of pedestrian images and their corresponding annotation files. Each annotation file contained bounding box coordinates, which were extracted using regular expressions and stored in the format (xmin, ymin, xmax, ymax). These coordinates were converted into PyTorch tensors and paired with additional data like labels, image identifiers, bounding box areas and crowd indicators forming a target dictionary for each image. The dataset was divided into training (80%) and testing (20%) parts to evaluate and visualise the model without any biases.
Preprocessing
All the images were converted to RGB format and transformed into tensors using the torchvision.transforms.functional API. Bounding boxes were kept as tensors to ensure compatibility with the Faster R-CNN model. The images were normalized according to the preprocessing pipeline of the resnet50 backbone, ensuring that the input is consistent.
To enhance training variability, random shuffling was applied to the training set. And since images contained a variable number of bounding boxes a custom collate function was made to correctly group together images and annotations during training.

These are the results after training the model for 10 epochs given the training data
Epoch 1: 
Average Loss: 0.3490
Epoch 2: 
Average Loss: 0.1269
Epoch 3: 
Average Loss: 0.0997
Epoch 4: 
Average Loss: 0.0800
Epoch 5: 
Average Loss: 0.0742
Epoch 6: 
Average Loss: 0.0718
Epoch 7: 
Average Loss: 0.0704
Epoch 8: 
Average Loss: 0.0697
Epoch 9: 
Average Loss: 0.0690
Epoch 10: 
Average Loss: 0.0691
This was the mAP evaluation of the model given the testing data
--- Evaluating on Test Set ---
map: 0.7985
map_50: 0.9998
map_75: 0.9382
map_small: -1.0000
map_medium: 0.8000
map_large: 0.8006
mar_1: 0.3728
mar_10: 0.8494
mar_100: 0.8494
mar_small: -1.0000
mar_medium: 0.8000
mar_large: 0.8513
map_per_class: -1.0000
mar_100_per_class: -1.0000
classes: 1.0000
