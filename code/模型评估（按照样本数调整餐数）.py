import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from matplotlib import rcParams
import re
import numpy as np
import nltk
from wordcloud import WordCloud
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# 指定 NLTK 数据的路径
nltk.data.path.append(r'C:\Users\李建辉\AppData\Roaming\nltk_data')
nltk.download('stopwords')

# 设置字体以支持中文显示
rcParams['font.sans-serif'] = ['Microsoft YaHei']
rcParams['axes.unicode_minus'] = False

# 数据准备
file_path = r'E:\机器学习\练习2\spam.csv'
data = pd.read_csv(file_path, encoding='latin-1')
data = data[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'message'})
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# 数据清洗
def clean_text(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', ' ', text)
    text = text.lower()
    return text

data['message'] = data['message'].apply(clean_text)

# 数据分割
X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42)

# 特征工程 - 使用 TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 计算类别权重
class_weights = {0: len(y_train) / (2 * (y_train == 0).sum()), 1: len(y_train) / (2 * (y_train == 1).sum())}

# 定义 Sigmoid 函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 定义代价函数
def compute_cost(h, y, theta, l2=0, class_weights=None):
    m = len(y)
    cost = -1/m * np.sum(y * np.log(h + 1e-15) + (1 - y) * np.log(1 - h + 1e-15))
    cost += (l2 / 2) * np.sum(np.square(theta))
    
    if class_weights is not None:
        # 类别加权
        weights = np.array([class_weights[label] for label in y])
        cost *= np.sum(weights)
        
    return cost

# 定义逻辑回归类
class LogisticRegressionManual:
    def __init__(self, alpha=0.01, iterations=5000, l2=0, class_weights=None):
        self.alpha = alpha
        self.iterations = iterations
        self.l2 = l2
        self.class_weights = class_weights
        self.theta = None
        self.J_history = []

    def fit(self, X, y):
        m, n = X.shape
        self.theta = np.zeros(n)

        for i in tqdm(range(self.iterations), desc='Training Progress'):
            z = np.dot(X, self.theta)
            h = sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / m
            if self.l2 > 0:
                gradient += self.l2 * self.theta
            self.theta -= self.alpha * gradient
            cost = compute_cost(h, y, self.theta, self.l2, self.class_weights)
            self.J_history.append(cost)

    def predict(self, X):
        z = np.dot(X, self.theta)
        h = sigmoid(z)
        return [1 if i >= 0.5 else 0 for i in h]

# 使用固定超参数训练模型
model = LogisticRegressionManual(alpha=0.01, iterations=5000, l2=0.1, class_weights=class_weights)
model.fit(X_train_tfidf.toarray(), y_train)
y_pred = model.predict(X_test_tfidf.toarray())

# 输出评估结果
print("模型评估指标：")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

# 可视化损失函数变化
plt.plot(range(len(model.J_history)), model.J_history)
plt.title('代价函数变化图（使用调整后的权重）')
plt.xlabel('迭代次数')
plt.ylabel('代价函数值')
plt.grid()
plt.show()

# 绘制混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('混淆矩阵（调整权重后模型）')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Ham', 'Spam'])
plt.yticks(tick_marks, ['Ham', 'Spam'])
thresh = cm.max() / 2.
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, cm[i, j], horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")
plt.ylabel('真实标签')
plt.xlabel('预测标签')
plt.show()

# 绘制ROC曲线
y_prob = sigmoid(np.dot(X_test_tfidf.toarray(), model.theta))
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC曲线 (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假阳性率')
plt.ylabel('真正率')
plt.title('接收者操作特性 ROC 曲线（调整权重后模型）')
plt.legend(loc="lower right")
plt.show()
