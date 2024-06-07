# Bank Note Denomination Classification
This is the repository for the Bank Note Denomination Classification project. The project is a part of the course work for the course "Computer Vision and Image Processing" at the University of Kwazulu-Natal. The project is aimed at classifying bank notes of different denominations using image processing techniques. The project is implemented in Python using the OpenCV library.

## Running and Configuring the Project
The notebook `train.ipynb` contains the code for training the machine learning model. The notebook is self-contained and can be run on any machine with Python installed. The notebook requires the libraries mentioned in the preamble of the notebook. The notebook can be run in any Python environment, such as Jupyter Notebook, Google Colab, or any other Python IDE, however the data should be located in the same directory as the notebook.

The notebook provides a configuration to change the feature extraction technique being used, by default we test each extraction technique with each model. The configuration can be changed by modifying the `descriptor` variable in the notebook. The machine learning model being evaluated can be adjusted by avoiding running the cells for the models that are not required.

We also provide an "inference" notebook `inference.ipynb` that can be used to test the model on new images. The notebook contains the code for loading the trained model and using it to classify new images. The notebook requires the same libraries as the training notebook. The notebook additionally requires the module "joblib" and "gradio" to be installed. The notebook can be run in any Python environment, such as Jupyter Notebook, Google Colab, or any other Python IDE. Gradio provides a web interface to test the model on new images. The notebook can be run by executing all the cells in the notebook, and following the link presented in the output of the last cell.


## Dataset
The dataset for the project is generated using images of bank notes of different denominations. The images are taken from online, and the dataset is created by cropping the images to extract the bank notes. The dataset also includes a variety of images taken by the participants of the project. We augment the dataset by applying various transformations to the images, such as rotation, scaling, and flipping. The dataset is divided into two parts: training data and testing data. The training data is used to train the machine learning model, and the testing data is used to test the model.


## Training the Model
The notebook `train.ipynb` contains the code for training the machine learning model. The notebook handles the entire process of training the model, including loading the training data, preprocessing the data, segmenting the bank notes, extracting features, and training the model. The notebook uses the OpenCV library to perform image processing tasks, such as edge detection, contour detection, and feature extraction. The notebook also uses the scikit-learn library to train the machine learning model.

We avoided the use of deep learning models due to the small size of the dataset. Instead, we used traditional machine learning models, such as Support Vector Machines (SVM) and Random Forests, to classify the bank notes. We experimented with different feature extraction techniques, such as Scale-Invariant Feature Transform (SIFT) and Oriented FAST and Rotated BRIEF (ORB), to extract features from the images. We also experimented with different machine learning models and hyperparameters to find the best model for the task.

## Testing the Model
The notebook `train.ipynb` contains the code for testing the machine learning model. The testing section is located towards the end of the notebook. The testing involves the process of processing the testing data, segmenting the bank notes, extracting features, and testing the model. The testing data is used to evaluate the performance of the model on unseen data. The model's performance is evaluated using metrics such as accuracy, precision, recall, and F1 score. We also evaluate the model's performance using a confusion matrix to visualize the model's predictions.
We do an evaluation for each model and feature extraction technique to determine the best model and feature extraction technique for the task.

## Results
The results of the project are presented in the form of a report. The report contains a detailed description of the project, including the dataset, the machine learning models, the feature extraction techniques, and the results. The report also includes the performance metrics of the models, such as accuracy, precision, recall, and F1 score. The report also includes a confusion matrix to visualize the model's predictions.
