import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score
import seaborn as sns



if __name__ == '__main__':
    testing_data = pd.read_csv(r'C:\Users\HP\PycharmProjects\pythonProject\dino_testing features.csv')
    testing_label = pd.read_csv(r'C:\Users\HP\PycharmProjects\pythonProject\b. IDRiD_Disease Grading_Testing Labels.csv')

    training_data = testing_data.drop(columns=['class'])
    training_label = testing_label['Retinopathy grade']

    kmeans_dino = KMeans(n_clusters=5, random_state=42,n_init=10)
    kmeans_dino.fit(testing_data)
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


def concat_data(data, label):
    return pd.concat([data, label], axis=1)


def read_data(path1, path2, path3, path4):
    training_data = pd.read_csv(path1)
    training_label = pd.read_csv(path2)
    training_label = training_label['Retinopathy grade']
    training_data_with_label = concat_data(training_data.iloc[:, 1:], training_label)

    training_data_with_label['Retinopathy grade'] = training_label
    testing_data = pd.read_csv(path3)
    testing_label = pd.read_csv(path4)

    testing_data = concat_data(testing_data.iloc[:, 1:], testing_label['Retinopathy grade'])
    testing_data['Retinopathy grade'] = testing_label['Retinopathy grade']

    return training_data_with_label, testing_data


def gaussian_probability(x, mu, sigma2):
    a = (1 / np.sqrt(2 * (np.pi) * sigma2))
    b = np.exp(-np.square(x - mu) / (2 * sigma2))
    return a * b


def predict(test_data, means, vars, prior_y):
    if isinstance(test_data, pd.DataFrame):
        test_data = test_data.values
    PX_y = gaussian_probability(test_data, means, vars)
    PX_y = np.prod(PX_y, axis=1)
    Py_X = PX_y * prior_y
    return np.argmax(Py_X)


if __name__ == '__main__':
    # read dino
    dino_train, dino_test = read_data("dino_features.csv", "a. IDRiD_Disease Grading_Training Labels.csv",
                                      "dino_testing features.csv", "b. IDRiD_Disease Grading_Testing Labels.csv")

    m = dino_train.shape[0]
    label_counts = dino_train['Retinopathy grade'].value_counts()
    prior_y = np.array(label_counts) / m

    mean_values = dino_train.groupby("Retinopathy grade").mean()
    var_values = dino_train.groupby("Retinopathy grade").var()
    dino_test_y = np.asarray(dino_test["Retinopathy grade"])
    means = np.asarray(mean_values)
    vars = np.asarray(var_values)
    predicted_labels = [predict(np.array(dino_test.iloc[i, :-1]), means, vars, prior_y) for i in
                        range(dino_test.shape[0])]

    # accuracy and confusion matrix
    accuracy = accuracy_score(dino_test_y, predicted_labels)
    conf_matrix = confusion_matrix(dino_test_y, predicted_labels)
    f1 = f1_score(dino_test_y, predicted_labels, average='weighted')

    # TP, TN, FP, FN
    TP = conf_matrix[1, 1]
    TN = conf_matrix[0, 0]
    FP = conf_matrix[0, 1]
    FN = conf_matrix[1, 0]

    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)

    print("DINO, Accuracy:", accuracy)
    print("DINO, Sensitivity:", sensitivity)
    print("DINO, Specificity:", specificity)
    print("DINO, F1-Score:", f1)
    print("DINO, Confusion Matrix:\n", conf_matrix)

    # clip
    clip_train, clip_test = read_data("clip_features.csv", "a. IDRiD_Disease Grading_Training Labels.csv",
                                      "clip_testing features.csv", "b. IDRiD_Disease Grading_Testing Labels.csv")

    m = clip_train.shape[0]
    label_counts = clip_train['Retinopathy grade'].value_counts()
    prior_y = np.array(label_counts) / m
    mean_values = clip_train.groupby("Retinopathy grade").mean()
    var_values = clip_train.groupby("Retinopathy grade").var()
    clip_test_y = np.asarray(clip_test["Retinopathy grade"])
    means = np.asarray(mean_values)
    vars = np.asarray(var_values)
    predicted_labels = [predict(np.array(clip_test.iloc[i, :-1]), means, vars, prior_y) for i in
                        range(clip_test.shape[0])]

    # accuracy and confusion matrix
    accuracy = accuracy_score(clip_test_y, predicted_labels)
    conf_matrix = confusion_matrix(clip_test_y, predicted_labels)

    f1 = f1_score(clip_test_y, predicted_labels, average='weighted')

    TP = conf_matrix[1, 1]
    TN = conf_matrix[0, 0]
    FP = conf_matrix[0, 1]
    FN = conf_matrix[1, 0]

    # Sensitivity and Specificity
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)

    print("CLIP, Accuracy:", accuracy)
    print("CLIP, Sensitivity:", sensitivity)
    print("CLIP, Specificity:", specificity)
    print("CLIP, F1-Score:", f1)
    print("CLIP, Confusion Matrix:\n", conf_matrix)



