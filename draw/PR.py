import pickle
import matplotlib.pyplot as plt
import numpy as np

file_paths = [
    "/Users/xuxiao/Downloads/pr_curve_data_vt_phq_3.pkl",
    "/Users/xuxiao/Downloads/pr_curve_data_vt_phq_4.pkl", 
    "/Users/xuxiao/Downloads/pr_curve_data_vt_phq_5.pkl", 
]

plt.figure(figsize=(8, 7))

# 为每个 pkl 文件生成一个唯一的颜色和线条样式
colors = ['#44A3BB', '#8F33CC', '#CC5233']
linestyles = ['-', '-', '-']

for i, file_path in enumerate(file_paths):
    with open(file_path, "rb") as f:
        pr_curve_data = pickle.load(f)

    recall, precision = pr_curve_data  # Unpack the tuple
    auc_score = abs(np.trapz(precision, recall))
    print(f"AUC for Curve {i+1}: {auc_score:.3f}")
    plt.plot(recall, precision, lw=3, color=colors[i % len(colors)], linestyle=linestyles[i % len(linestyles)], marker='.', label=f"PHQ {i+3} (AUC: {auc_score:.3f})")

plt.title('Precision-Recall curve', fontsize=20, fontname='Arial')
plt.xlabel('Recall', fontsize=18, fontname='Arial')
plt.ylabel('Precision', fontsize=18, fontname='Arial')
plt.grid(True)
plt.legend(loc='lower right', fontsize=18, title_fontsize='18')  # Add a legend at lower right corner
plt.savefig('345.jpg', dpi=600, bbox_inches='tight')
