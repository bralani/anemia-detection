# Repository anemia-detection

MIT License
Copyright (c) [2023] [Balice Matteo]

Welcome to the anemia-detection repository! This repository contains code and related resources for the process of detecting anemia through ocular images.

## Repository Contents

The repository is organized into four main folders:

1. **database_sclere**: This folder contains the dataset of ocular images used for training and evaluating sclera segmentation models. The "RAW" images are 218 and organized into subfolders to facilitate data access and management.

2. **sclera_segmentation**: In this folder, you will find the code and scripts necessary to perform sclera segmentation.

3. **svm**: It contains all image preprocessing and the final svm model.

4. **unet**: It contains the deep pre-trained model with preprocessing functions.

5. **entire_workflow.ipynb**: contains the entire project workflow combined into a single file (from sclera segmentation to final prediction).

6. **generate_vessel_masks.ipynb**: contains the code for segmenting scleral vessels in eye images using a convolutional neural network (U-Net) model. After preprocessing the images, the code loops through a DataFrame (df) containing patient information. For each row of the DataFrame, scleral vessel segmentation is performed on the corresponding image using the pretrained U-Net model.

7. **density_evaluation.ipynb**: contains a script that defines a function to calculate and print various evaluation scores for a classification model. Scores calculated include: f1, f2, precision, recall, accuracy, and roc_auc.

8. **preprocessing.ipynb**: contains a script that defines some image preprocessing functions and then applies these functions to a set of training images.

9. **sclera_segmentation.ipynb**: shows step by step automatic sclera segmentation.

## Installation and Usage

To get started, follow these instructions for installing and using the code and resources in the repository:

1. Clone this repository to your local system using the command:
```git clone https://github.com/bralani/anemia-detection.git```

2. Follow the workflow in the entire_workflow.ipynb file.

## License

Please refer to the [LICENSE](LICENSE) file for more information on the rights and restrictions related to the use, modification, and distribution of this software.

## Support

For any questions or issues, you can open an issue in the Issue section of this repository. We will do our best to respond as quickly as possible.
