# Face Mask Detection using VGG16 Transfer Learning

## Project Overview

Face Mask Detection is a Computer Vision and Deep Learning project developed to automatically identify whether a person is wearing a face mask or not. The system can be used in public places, educational institutions, healthcare facilities, and workplaces to monitor compliance with safety regulations.

The application performs real-time face detection and classifies each detected face into one of two categories:

* With Mask
* Without Mask

## Dataset

This project was trained using the **Face Mask 12K Images Dataset** from Kaggle, containing approximately 12,000 images divided into two classes:

* With Mask
* Without Mask

The dataset includes facial images captured under different lighting conditions, viewing angles, backgrounds, and mask variations, enabling the model to learn robust and generalized features.

### Data Preprocessing

The following preprocessing techniques were applied before training:

* Image resizing
* Pixel normalization
* Random horizontal flipping
* Rotation augmentation
* Zoom augmentation
* Dataset shuffling

These techniques improved the model's ability to generalize to unseen images and reduced overfitting.

## Transfer Learning with VGG16

Training a deep neural network from scratch requires a large dataset and significant computational resources. To overcome this challenge, **VGG16** pre-trained on the ImageNet dataset was used as the backbone model.

Initially, the convolutional layers of VGG16 were loaded with pre-trained ImageNet weights and used as a feature extractor. The original classification head was removed and replaced with custom layers designed specifically for face mask classification.

### Custom Classification Head

* Global Average Pooling Layer
* Dropout Layer
* Dense Output Layer (Sigmoid Activation)

## Fine-Tuning

After training the custom classification layers, the upper layers of VGG16 were unfrozen and fine-tuned using a low learning rate. This allowed the model to adapt ImageNet features to the face mask detection task while preserving previously learned visual representations.

Fine-tuning resulted in:

* Improved classification accuracy
* Better feature extraction
* Faster convergence
* Enhanced real-world performance

## Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Streamlit
* VGG16 Transfer Learning

## Applications

* Public Safety Monitoring
* Healthcare Facilities
* Educational Institutions
* Offices and Workplaces
* Transportation Hubs

## Future Improvements

* Multi-face tracking
* Mobile deployment
* Cloud-based monitoring
* Mask compliance analytics dashboard

## Conclusion

This project demonstrates the effectiveness of Transfer Learning and Fine-Tuning for real-world Computer Vision applications. By leveraging VGG16 and the Face Mask 12K Images Dataset, the system achieves accurate and efficient face mask classification while maintaining low training costs and reduced development time.
