# Human-Emotions-Dataset
 Human Detection and Emotion Recognition with ViT, CNN, and Quantization for resource-efficient AI. Explore computer vision techniques.
# My Emotion Detection Model

## References (Continued)
- [My-Model](https://github.com/Ammar-Abdelhady-ai/Human-Emotions-Dataset/blob/main/My-Model-1.ipynb)
- [Model Visualization, Patch Encoder, and Transfer Learning](https://github.com/Ammar-Abdelhady-ai/Human-Emotions-Dataset/blob/main/Model%20Visualization%20%2CPatch%20Encoder%20and%20Transfer%20Learning-2.ipynb)
- [Hugging Face Model with WandB](https://github.com/Ammar-Abdelhady-ai/Human-Emotions-Dataset/blob/main/Hugging%20Face%20Model%20with%20WandB-4.ipynb)
- [Cutmix Augmentation and TensorFlow Record](https://github.com/Ammar-Abdelhady-ai/Human-Emotions-Dataset/blob/main/Cutmix%20Augmentation%20and%20Tensorflow%20Record-3.ipynb)
- [ONNX Quantization and TensorFlow Lite Model Accuracy Evaluation and Conversion](https://github.com/Ammar-Abdelhady-ai/Human-Emotions-Dataset/blob/main/ONNX%20Quantization%20and%20TensorFlow%20Lite%20Model%20Accuracy%20Evaluation%20and%20Conversion-5.ipynb)
- [APl](https://github.com/Ammar-Abdelhady-ai/Human-Emotions-Dataset/tree/main/api%20env)
- [Data](https://www.kaggle.com/datasets/muhammadhananasghar/human-emotions-datasethes)

---

## Visualizing Model Features, GradCAM, and Patch Encoding

In this file, we explore various aspects related to deep learning models, feature visualization, and image processing. The code in this file covers the following key sections:

### Feature Map Visualization

- We use a VGG16 backbone to extract feature maps from the model.
- The code defines a custom function to identify convolutional layers and extract feature maps.
- Feature maps are visualized for inspection and analysis.

### GradCAM (Gradient-weighted Class Activation Mapping)

- We leverage an EfficientNetB5 model for GradCAM, a technique for visualizing which parts of an image are important for predictions.
- The code implements GradCAM to highlight regions that influence model predictions.

### Patch Encoding

- We extract patches from an image using TensorFlow's image patch extraction utility.
- The patches are visualized for better understanding.

### Custom Residual Network (ResNet34)

- We define a custom ResNet34 architecture.
- This architecture includes custom convolutional layers and residual blocks for deep feature extraction.

These sections offer insights into feature visualization, model interpretability, and custom network design. You can use this code as a reference or for educational purposes, adapting it to your specific deep learning tasks and projects.

---

## Hugging Face Vision Transformer (ViT) for Computer Vision

This code demonstrates the usage of Hugging Face's Transformers library to work with Vision Transformer (ViT) models for computer vision tasks. Vision Transformers have emerged as powerful models in the field of computer vision, and Hugging Face simplifies the access and fine-tuning of ViT models.

### Model Configuration Initialization

- The code starts by initializing a ViT model configuration, specifying key parameters like hidden size and dropout probability.

### Loading Pre-trained ViT Model

- A pre-trained ViT model (google/vit-base-patch16-224-in21k) is loaded from Hugging Face's model repository. This model has been pre-trained on a large image dataset and can be fine-tuned for specific computer vision tasks.

### Custom Model Architecture

- A custom model architecture is defined using TensorFlow. The input image size is (256, 256, 3), and it's processed through the pre-trained ViT model. Additional layers, including a dense layer with softmax activation, are added for classification.

### Weights and Biases (WandB) Setup

- The code configures Weights and Biases (WandB) to track experiments. This allows for easy monitoring and logging of training progress, hyperparameters, and results.

### Custom Callbacks

- Two custom callbacks, LogConfMatrix and LogResultsTable, are defined. LogConfMatrix logs the confusion matrix for model evaluation, while LogResultsTable logs a table of model predictions and labels.

### Model Training

- The ViT-based model is trained using the fit method. The code specifies the number of epochs and other hyperparameters for training.

### Model description

- Model Link: google/vit-base-patch16-224-in21k
- The Vision Transformer (ViT) is a transformer encoder model (BERT-like) pretrained on a large collection of images in a supervised fashion, namely ImageNet-21k, at a resolution of 224x224 pixels.

- Images are presented to the model as a sequence of fixed-size patches (resolution 16x16), which are linearly embedded. One also adds a [CLS] token to the beginning of a sequence to use it for classification tasks. One also adds absolute position embeddings before feeding the sequence to the layers of the Transformer encoder.

---

## Data Augmentation and TensorFlow Records

This code snippet demonstrates various techniques for data augmentation, specifically the CutMix technique, and the creation of TensorFlow Records for efficient data storage and retrieval. Data augmentation is crucial for enhancing the robustness of machine learning models by generating diverse training data.

### Data Augmentation Layers

- The code begins by defining a set of data augmentation layers using TensorFlow's Keras API. These layers include:
  - RandomRotation: Randomly rotates images within a specified range.
  - RandomFlip: Randomly flips images horizontally.
  - RandomContrast: Applies random contrast adjustments to images.

### Augmentation Function

- An augment_layer function is defined, which applies the previously defined data augmentation layers to an image. It sets the training mode to True to indicate that data augmentation should occur during training.

### Bounding Box Calculation

- The box function calculates the coordinates of a bounding box based on a parameter called lambda. The bounding box is defined by its top-left corner coordinates (r_x and r_y) and its dimensions (r_h and r_w).

### Label Type Conversion

- The fit_label function simply converts a label to the tf.float32 data type, which can be useful for machine learning tasks.

### CutMix Data Augmentation

- The cutmix function applies the CutMix data augmentation technique. It combines parts of two input images, image_1 and image_2, to create a new image, and it calculates a new label based on the combination.

### Dataset Augmentation

- Two datasets, train_dataset_1 and train_dataset_2, are created by applying the augment_layer function to a base dataset, typically used for training a machine learning model.

- The code then combines these two augmented datasets using the zip function to create a mixed_dataset. The cutmix function is applied to this mixed dataset to generate a new dataset, training_data, containing augmented images and labels.

### TensorFlow Record Creation

- The code also demonstrates how to create TensorFlow Records, which are a binary format for efficient storage and retrieval of data, commonly used in TensorFlow for large datasets.

- The TensorFlow Records are generated by encoding images and labels, then writing them to separate shard files. This process is orchestrated using loops and TensorFlow's TFRecordWriter.

- Finally, the code

### Deployment Using FastAPI and ONNX Quantized Model

We have deployed our emotion detection model using FastAPI, which allows you to make predictions via an API. Additionally, we have optimized our model using ONNX quantization for improved efficiency and performance.

## Face Recognition Feature in Harmony with Emotion Detection

Our ongoing commitment to enhancing the capabilities of our emotion detection model has led to the introduction of a new Face Recognition Feature. This feature seamlessly integrates with our existing Human Emotion Detection system, providing an enriched analysis of images.

### How it Complements Emotion Detection

1. **Emotion Context:**
   - The Face Recognition Feature adds an additional layer of context to emotion detection by identifying and recognizing faces within images.

2. **User Engagement:**
   - Recognizing faces allows for a more personalized analysis of emotions, considering individual expressions and reactions.

3. **Security and Confirmation:**
   - The integration of face identification not only enhances the emotional analysis but also introduces a layer of security. Users are prompted to confirm the detected face, ensuring accuracy and user engagement.

### Seamless Integration

The Face Recognition Feature seamlessly integrates into our Human Emotion Detection pipeline. Whether you are predicting emotions in images or exploring deep learning models, the face recognition component enhances the overall analysis.

### Usage in Conjunction with Emotion Detection

1. **Combined Analysis:**
   - Run your existing Emotion Detection scripts or applications as usual, and now enjoy the added insights provided by face recognition.

2. **Extended Reports:**
   - The reports generated by your Emotion Detection system will now include additional information about recognized faces.

3. **Unified API:**
   - If you're using our API, the Face Recognition Feature is seamlessly integrated, providing a unified interface for both emotion detection and face recognition.

Feel free to explore the combined power of emotion detection and face recognition in your applications, and don't hesitate to provide feedback or report any issues.

## Acknowledgments

We appreciate the continued support from the open-source community, particularly those contributing to OpenCV, which has been instrumental in the implementation of the Face Recognition Feature.

---

