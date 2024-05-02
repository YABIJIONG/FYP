# Project Overview

This project focuses on feature extraction and classification using different combinations of models and algorithms. Each script in the repository is designed to preprocess data, apply normalization, and conduct clustering or classification:

1. **Clip+Guassian.py**: Utilizes the CLIP model for feature extraction. After feature extraction, it standardizes the features to fit a Gaussian distribution (z-score normalization) and classifies the data using the Naive Bayes algorithm.
2. **Clip+Kmeans.py**: Employs the CLIP model for feature extraction, followed by clustering using K-means.
3. **dino+Guassian.py**: Uses the DINO model for feature extraction, normalizes the features to a Gaussian distribution, and classifies using the Naive Bayes algorithm.
4. **dino+k-means.py**: Applies the DINO model for feature extraction and performs clustering using K-means.

## Requirements

To run the scripts, ensure your environment is set up with the following:

- Python 3.9
- matplotlib 3.8.3
- numpy 1.26.2
- seaborn 0.13.2

**Note:** The sklearn package was upgraded during the project. Accuracy, F1-Score, and Sensitivity were computed using sklearn version 1.3.2, while Specificity was calculated using sklearn version 1.1.3.

## Setup and Execution

Update the paths in the scripts to your local data directories before running them:

```python
training_data = pd.read_csv(r'YOUR_LOCAL_PATH\clip_testing_features.csv')
training_label = pd.read_csv(r'YOUR_LOCAL_PATH\b. IDRiD_Disease Grading_Testing Labels.csv')
```python
Replace YOUR_LOCAL_PATH with the actual directory where your data files are stored.

## Running the Scripts
Execute the scripts from your command line by navigating to their directory:
python Clip+Guassian.py
Make sure you activate the correct Python environment where all dependencies have been installed.

## Contribution
Contributions are welcome. You can contribute by submitting pull requests with bug fixes, enhancements, or new features.
