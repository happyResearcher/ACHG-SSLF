import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Example data (replace with your actual data)
classifiers = ['c_LR', 'c_NN', 'c_RF', 'c_SVM']
features = ['Nontopo', 'Topo', 'NodeEmb', 'NodeEmb+']

macro_f1 = np.array([
    [61.02, 62.83, 63.93, 64.35],
    [60.38, 61.39, 69.28, 71.99],
    [70.64, 73.61, 76.02, 76.25],
    [61.04, 35.21, 63.42, 64.17]
])

minority_f1 = np.array([
    [59.97, 61.31, 63.29, 63.84],
    [57.27, 59.74, 69.46, 73.09],
    [70.08, 73.51, 75.99, 76.23],
    [59.92, 66.69, 62.41, 63.54]
])

bar_width = 0.2
index = np.arange(len(classifiers))

# Define a modified colormap with increased contrast for the first color
cmap_colors = plt.cm.YlGnBu(np.linspace(0, 1, len(features)))
cmap_colors[0, :] = [0.2, 0.2, 0.6, 1.0]  # Darker blue for the first color
modified_cmap = LinearSegmentedColormap.from_list('ModifiedYlGnBu', cmap_colors, N=len(features))

# Create subplots for Macro F1 and Minority F1
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

# Plot Macro F1
for i, feature in enumerate(features):
    ax1.bar(index + i * bar_width, macro_f1[:, i], width=bar_width, label=feature, color=modified_cmap(i / len(features)))

ax1.set_xlabel('Classifier')
ax1.set_ylabel('Macro average F1 score (%)')
ax1.set_xticks(index + bar_width * (len(features) - 1) / 2)
ax1.set_xticklabels(classifiers)
ax1.legend(loc='upper left')
ax1.set_ylim(bottom=30, top=90)

# Plot Minority F1
for i, feature in enumerate(features):
    ax2.bar(index + i * bar_width, minority_f1[:, i], width=bar_width, label=feature, color=modified_cmap(i / len(features)))

ax2.set_xlabel('Classifier')
ax2.set_ylabel('Negative class F1 score (%)')
ax2.set_xticks(index + bar_width * (len(features) - 1) / 2)
ax2.set_xticklabels(classifiers)
ax2.legend(loc='upper left')
ax2.set_ylim(bottom=50, top=90)

plt.tight_layout()
plt.savefig('./data/' + 'balancedF1Comparison', dpi=600)
plt.show()
