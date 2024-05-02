import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 加载数据
training_data = pd.read_csv(r'C:\Users\HP\PycharmProjects\pythonProject\clip_testing features.csv')
training_labels = pd.read_csv(r'C:\Users\HP\PycharmProjects\pythonProject\b. IDRiD_Disease Grading_Testing Labels.csv')

# 准备数据
training_data = training_data.drop(columns=['Class'])
training_labels = training_labels['Retinopathy grade']

# 创建高斯朴素贝叶斯分类器实例
gnb = GaussianNB()

# 训练模型
gnb.fit(training_data, training_labels)

# 预测标签
predicted_labels = gnb.predict(training_data)

# 计算和打印性能指标
accuracy = accuracy_score(training_labels, predicted_labels)
recall = recall_score(training_labels, predicted_labels, average='macro')
f1 = f1_score(training_labels, predicted_labels, average='macro')

print("Accuracy:", accuracy)
print("Recall:", recall)
print("F1-Score:", f1)

# 生成并打印混淆矩阵
conf_matrix = confusion_matrix(training_labels, predicted_labels)
print(conf_matrix)
labels = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4']
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix Heatmap')
plt.show()


specificities = []
for i in range(5):
    TN = np.sum(conf_matrix) - np.sum(conf_matrix[i, :]) - np.sum(conf_matrix[:, i]) + conf_matrix[i, i]
    FP = np.sum(conf_matrix[:, i]) - conf_matrix[i, i]
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    specificities.append(specificity)

print("Specificities:", sum(specificities)/5)
