# 导入必要的库
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

# 确保下载nltk的停用词
nltk.download('stopwords')

# 设置字体以支持中文显示
rcParams['font.sans-serif'] = ['Microsoft YaHei']
rcParams['axes.unicode_minus'] = False

# 数据准备
file_path = r'E:\机器学习\练习2\spam.csv'
data = pd.read_csv(file_path, encoding='latin-1')
data = data[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'message'})
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

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

# 定义 Sigmoid 函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 定义代价函数
def compute_cost(h, y, theta, l2=0):
    m = len(y)
    cost = -1/m * np.sum(y * np.log(h + 1e-15) + (1 - y) * np.log(1 - h + 1e-15))
    cost += (l2 / 2) * np.sum(np.square(theta))
    return cost

# 定义逻辑回归类
class LogisticRegressionManual:
    def __init__(self, alpha=0.01, iterations=5000, l2=0):
        self.alpha = alpha
        self.iterations = iterations
        self.l2 = l2
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
            cost = compute_cost(h, y, self.theta, self.l2)
            self.J_history.append(cost)

    def predict(self, X):
        z = np.dot(X, self.theta)
        h = sigmoid(z)
        return [1 if i >= 0.5 else 0 for i in h]

    def search_optimal_learning_rate(self, X, y, learning_rates):
        best_alpha = None
        best_cost = float('inf')
        cost_history = []

        for alpha in learning_rates:
            self.alpha = alpha
            self.J_history = []
            self.fit(X, y)
            final_cost = self.J_history[-1]
            cost_history.append(final_cost)
            print(f"Learning rate: {alpha}, Final cost: {final_cost}")

            if final_cost < best_cost:
                best_cost = final_cost
                best_alpha = alpha

        self.alpha = best_alpha
        print(f"Optimal learning rate: {best_alpha} with cost: {best_cost}")
        return best_alpha, cost_history

# 搜索最优学习率
model = LogisticRegressionManual(iterations=1000, l2=0)
learning_rates = [0.001, 0.01, 0.05, 0.1, 0.5]
optimal_alpha, cost_history = model.search_optimal_learning_rate(X_train_tfidf.toarray(), y_train.values, learning_rates)

# 使用最优学习率重新训练模型
model = LogisticRegressionManual(alpha=optimal_alpha, iterations=5000, l2=0)
model.fit(X_train_tfidf.toarray(), y_train)
y_pred = model.predict(X_test_tfidf.toarray())

# 输出评估指标
print("模型评估指标：")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

# 可视化损失函数的变化
plt.plot(range(len(model.J_history)), model.J_history)
plt.title('代价函数变化图 - 最优学习率')
plt.xlabel('迭代次数')
plt.ylabel('代价函数值')
plt.grid()
plt.show()

# 绘制混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('混淆矩阵')
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
plt.title('ROC曲线')
plt.legend(loc="lower right")
plt.show()
