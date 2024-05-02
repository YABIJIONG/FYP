# Project Overview

This project focuses on feature extraction and classification using different combinations of models and algorithms. Each script in the repository is designed to preprocess data, apply normalization, and conduct clustering or classification:

**Clip+Guassian.py**: Feature extraction using Clip model. After feature extraction, Gaussian model was used to classify the data.
2. **Clip+Kmeans.py**: Clip model is used for feature extraction, and then K-means is used to classify the data.
3. **dino+Guassian.py**: dino model was used for feature extraction and Gaussian model for classification.
4. **dino+ K-means. py**: dino model was used for feature extraction and K-means for classification.

## Requirements

To run the scripts, ensure your environment is set up with the following:

- Python 3.9
- matplotlib 3.8.3
- numpy 1.26.2
- seaborn 0.13.2

**Note:** The sklearn package was upgraded during the project. Accuracy, F1-Score, and Sensitivity were computed using sklearn version 1.3.2, while dino with k-means Specificity was calculated using sklearn version 1.1.3.

## Setup and Execution

Update the paths in the scripts to your local data directories before running them:

```python
training_data = pd.read_csv(r'YOUR_LOCAL_PATH\clip_testing_features.csv')
training_label = pd.read_csv(r'YOUR_LOCAL_PATH\b. IDRiD_Disease Grading_Testing Labels.csv')
```

Replace YOUR_LOCAL_PATH with the actual directory where your data files are stored.

## Running the Scripts
Execute the scripts from your command line by navigating to their directory:
python Clip+Guassian.py
Make sure you activate the correct Python environment where all dependencies have been installed.

## Contribution
Contributions are welcome. You can contribute by submitting pull requests with bug fixes, enhancements, or new features.
