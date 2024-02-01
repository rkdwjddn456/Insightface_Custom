import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# 0: incorrect
# 1: correct
y_true = np.array([0, 1, 1, 1, 1, 1, 0, 0, 0, 0,
                   1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 
                   0, 0, 0, 1, 1, 1, 1, 1, 0, 1,
                   1, 1, 1, 0, 1, 0, 0, 1, 1, 1,
                   0, 1, 0, 0, 1, 1, 1, 1, 0, 1,
                   1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 
                   0, 0, 1, 1]) 

# cosine_similarity
y_scores = np.array([0.26526988, 0.34894228, 0.3543207, 0.31759936, 0.23755816, 0.2707153,
                0.2162132, 0.22614294, 0.2299712, 0.36639529, 0.17418104, 0.46965656,
                0.4541471, 0.08869932, 0.37138259, 0.18836826, 0.50829226, 0.36123943,
                0.30643538, 0.31273133, 0.14356959, 0.29708257, 0.20915657, 0.27283442,
                0.35353947, 0.25264645, 0.25314656, 0.15177777, 0.29767889, 0.39289993,
                0.31518489, 0.36692107, 0.27130657, 0.31368881, 0.21218178, 0.12604778,
                0.2497893, 0.36714706, 0.36139953, 0.31416875, 0.23014905, 0.27671334,
                0.24766734, 0.15967268, 0.20298743, 0.36916178, 0.28902566, 0.41194913,
                0.19642088, 0.52988809, 0.25057995, 0.27432993, 0.18286943, 0.12249452,
                0.2583819, 0.39452732, 0.18018034, 0.34757894, 0.1707263, 0.25249243,
                0.12745225, 0.27874923, 0.35758719, 0.36429912])

fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

precision, recall, _ = precision_recall_curve(y_true, y_scores)

f1_scores = 2 * (precision * recall) / (precision + recall)

max_f1_score = max(f1_scores)
print("Maximum F1 Score:", max_f1_score)

best_threshold = thresholds[np.argmax(f1_scores)]
print("Best Threshold for Maximum F1 Score:", best_threshold)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")

plt.subplot(1, 2, 2)
plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")

plt.axhline(y=max_f1_score, color='r', linestyle='-', label='Max F1 Score(= %0.2f)' % max_f1_score)
plt.axvline(x=recall[np.argmax(f1_scores)], color='g', linestyle='--', label='Best Threshold for Max F1 Score(= %0.2f)' % best_threshold)
plt.legend()

plt.tight_layout()
plt.show()