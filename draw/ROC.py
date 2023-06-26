import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.metrics import auc
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve

# 单独加载每一个文件
with open("/Users/xuxiao/Downloads/roc_curve_data_envelope.pkl", "rb") as f:
    roc_curve1 = pickle.load(f)
with open("/Users/xuxiao/Downloads/roc_curve_data_spectrogram.pkl", "rb") as f:
    roc_curve2 = pickle.load(f)
with open("/Users/xuxiao/Downloads/roc_curve_data_melspectrogram.pkl", "rb") as f:
    roc_curve3 = pickle.load(f)
with open("/Users/xuxiao/Downloads/roc_curve_data_emolarge.pkl", "rb") as f:
    roc_curve4 = pickle.load(f)
with open("/Users/xuxiao/Downloads/roc_curve_data_fusion.pkl", "rb") as f:
    roc_curve5 = pickle.load(f)

roc_curves = [roc_curve1, roc_curve2, roc_curve3, roc_curve4]

base_fpr = np.linspace(0, 1, 101)

roc_curves_means = []

for roccurve in roc_curves:
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    for i, (val_true, val_probs) in enumerate(roccurve):
        fpr, tpr, _ = roc_curve(val_true, val_probs)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        tprs.append(np.interp(mean_fpr, fpr, tpr))

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    roc_curves_means.append((mean_fpr, mean_tpr))

roc_curves_means.append(roc_curve5[-1])

labels = [
    "Upper Envelope",
    "Spectrogram",
    "Mel Spectrogram",
    "HSFs",
    "Fusion(Area : 0.847 ± 0.044)",
]

# 设置字体
font = {'family': 'Arial',
        'weight': 'normal',
        'size': 18,
        }
plt.rc('font', **font)

# 画ROC曲线
plt.figure(figsize=(8, 7))
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')  # 45度线
for i, (fpr, tpr) in enumerate(roc_curves_means):
    if i == 4:  # 第五个曲线加粗
        plt.plot(fpr, tpr, lw=4, label=labels[i])
    else:
        plt.plot(fpr, tpr, lw=2, label=labels[i])

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=16, fontname="Arial")
plt.ylabel('True Positive Rate', fontsize=16, fontname="Arial")
plt.title('Receiver Operating Characteristic Curves', fontsize=20)
plt.legend(loc="lower right", prop={'family':'Arial', 'size':16})
plt.show()
