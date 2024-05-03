# Project Overview

This project focuses on feature extraction and classification using different combinations of models and algorithms. Each script in the repository is designed to preprocess data, apply normalization, and conduct clustering or classification:

1.**Clip+Guassian.py**: Feature extraction using CLIP model. After feature extraction, Gaussian model is used to classify the data.
2.**Clip+Kmeans.py**: CLIP model is used for feature extraction, and then K-means is used to classify the data.
3.**dino+Guassian.py**: DINOv2 model is used for feature extraction and Gaussian model for classification.
4.**dino+ K-means. py**: DINOv2 model is used for feature extraction and K-means for classification.
5.**DINO PCA. py**: The DINOv2 model is used for feature extraction, PCA is used for dimensionality reduction and K-eans cluster analysis
6.**CLIP PCA. py**: The CLIP model is used for feature extraction, PCA is used for dimensionality reduction and K-eans cluster analysis


## Requirements

To run the scripts, ensure your environment is set up with the following:

- Python 3.9
- matplotlib 3.8.3
- numpy 1.26.2
- seaborn 0.13.2
- pandas 2.1.4

**Note:** The sklearn package was upgraded during the project. Accuracy, F1-Score, and Sensitivity were computed using sklearn version 1.3.2, while dino with k-means Specificity was calculated using sklearn version 1.1.3.

## Setup and Execution

Update the paths in the scripts to your local data directories before running them:

```python
training_data = pd.read_csv(r'YOUR_LOCAL_PATH\clip_testing_features.csv')
training_data = pd.read_csv(r'YOUR_LOCAL_PATH\dino_testing_features.csv')
training_label = pd.read_csv(r'YOUR_LOCAL_PATH\b. IDRiD_Disease Grading_Testing Labels.csv')
features = pd.read_csv(r'YOUR_LOCAL_PATH\clip_features.csv')
features = pd.read_csv(r'YOUR_LOCAL_PATH\dino_features.csv')
labels = pd.read_csv(r'YOUR_LOCAL_PATH\a. IDRiD_Disease Grading_Training Labels.csv')
```

Replace YOUR_LOCAL_PATH with the actual directory where your data files are stored.

## Running the Scripts
Execute the scripts from your command line by navigating to their directory:
python Clip+Guassian.py
Make sure you activate the correct Python environment where all dependencies have been installed.

## Contribution
Contributions are welcome. You can contribute by submitting pull requests with bug fixes, enhancements, or new features.

## Results
### Table 1: Results of Multivariate Classification Feature Data Sets Using k-means Model

| Model  | Accuracy | Sensitivity | Specificity | F1-score |
|--------|----------|-------------|-------------|----------|
| DINOv2 | 0.3      | 0.244       | 0.790       | 0.272    |
| CLIP   | 0.233    | 0.178       | 0.81        | 0.154    |

### Table 2: Results of Multivariate Classification Feature Data Sets Using Gaussian Model

| Model  | Accuracy | Sensitivity | Specificity | F1-score |
|--------|----------|-------------|-------------|----------|
| DINOv2 | 0.835    | 0.895       | 0.954       | 0.858    |
| CLIP   | 0.767    | 0.779       | 0.934       | 0.782    |

### Confusion Matrix: Gaussian and k-means Model for CLIP and DINOv2

#### CLIP with Gaussian Model

|     | C1  | C2 | C3 | C4 | C5 |
|-----|-----|----|----|----|----|
| C1  | 32  |  0 |  2 |  0 |  0 |
| C2  |  0  |  5 |  0 |  0 |  0 |
| C3  | 10  |  1 | 20 |  1 |  0 |
| C4  |  4  |  0 |  0 | 15 |  0 |
| C5  |  3  |  0 |  0 |  3 |  7 |

#### CLIP with k-means Model

|     | C1  | C2 | C3 | C4 | C5 |
|-----|-----|----|----|----|----|
| C1  | 16  | 11 |  1 |  0 |  6 |
| C2  |  2  |  1 |  0 |  0 |  2 |
| C3  |  8  | 10 |  7 |  0 |  7 |
| C4  |  8  |  5 |  6 |  0 |  0 |
| C5  |  4  |  2 |  6 |  1 |  0 |

#### DINOv2 with Gaussian Model

|     | C1  | C2 | C3 | C4 | C5 |
|-----|-----|----|----|----|----|
| C1  | 33  |  0 |  1 |  0 |  0 |
| C2  |  0  |  5 |  0 |  0 |  0 |
| C3  |  8  |  0 | 22 |  1 |  1 |
| C4  |  1  |  0 |  1 |  1 | 11 |

#### DINOv2 with k-means Model

|     | C1  | C2 | C3 | C4 | C5 |
|-----|-----|----|----|----|----|
| C1  | 18  |  1 | 15 |  0 |  0 |
| C2  |  3  |  0 |  2 |  0 |  0 |
| C3  | 13  | 11 |  3 |  4 |  1 |
| C4  |  1  |  3 |  5 |  7 |  3 |
| C5  |  0  |  4 |  4 |  2 |  3 |

DNO PCA
![DINO PCA](https://github.com/YABIJIONG/FYP/assets/168736842/755633e8-6a5d-491b-a7ca-5416f37bd760)

CLIP PCA
![CLIP PCA](https://github.com/YABIJIONG/FYP/assets/168736842/d239b82a-046e-4466-9766-905bf0b8517f)
