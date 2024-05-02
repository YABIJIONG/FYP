import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score
import seaborn as sns

# def concat_data(data, label):
#     return pd.concat([data, label], axis=1)
#
#
# def read_data(path1, path2):
#     training_data = pd.read_csv(path1)
#     training_label = pd.read_csv(path2)
#
#     training_label = training_label['Retinopathy grade']
#     print(training_label)
#
#     return training_label, training_data


if __name__ == '__main__':
    training_data = pd.read_csv(r'C:\Users\HP\PycharmProjects\pythonProject\dino_features.csv')
    training_label = pd.read_csv(r'a. IDRiD_Disease Grading_Training Labels.csv')

    training_data = training_data.drop(columns=['Class'])
    training_label = training_label['Retinopathy grade']

    kmeans_dino = KMeans(n_clusters=5, random_state=42,n_init=10)
    kmeans_dino.fit(training_data)
    dino_labels = kmeans_dino.labels_

    accuracy = accuracy_score(training_label, dino_labels)
    print("Accuracy:", accuracy)

    recall = recall_score(training_label, dino_labels,average='macro')
    print("Recall:",recall)


# # 计算F1分数
    f1 = f1_score(training_label, dino_labels, average='macro')
    print("F1-Score:", f1)

# 生成并打印混淆矩阵
    conf_matrix = confusion_matrix(training_label, dino_labels)
    labels = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4']
    print("Confusion Matrix:\n", conf_matrix)
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)  # 设置字体大小
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix Heatmap')
    plt.show()
# 计算 specificiy
TN = conf_matrix[0, 0]  # True Negative
FP = conf_matrix[0, 1]  # False Positive

specificity = TN / (TN + FP)
print("Specificity:", specificity)
