import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Example data (replace with your actual data)
minority_ratios = [0.3, 0.1, 0.0159]
features = ['Nontopo', 'Topo', 'NodeEmb', 'NodeEmb+']

macro_f1 = np.array([
    [68.41, 71.56, 73.92, 73.83],
    [61.49, 62.28, 62.90, 62.92],
    [51.48, 52.15, 51.95, 52.01]
])

minority_f1 = np.array([
    [52.07, 57.43, 61.34, 61.21],
    [28.69, 30.21, 31.39, 31.44],
    [3.97, 5.32, 4.88, 5.00]
])

bar_width = 0.2
index = np.arange(len(minority_ratios))

# Define a modified colormap with increased contrast for the first color
cmap_colors = plt.cm.YlGnBu(np.linspace(0, 1, len(features)))
cmap_colors[0, :] = [0.2, 0.2, 0.6, 1.0]  # Darker blue for the first color
modified_cmap = LinearSegmentedColormap.from_list('ModifiedYlGnBu', cmap_colors, N=len(features))

# Create subplots for Macro F1 and Minority F1
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

# Plot Macro F1
for i, feature in enumerate(features):
    ax1.bar(index + i * bar_width, macro_f1[:, i], width=bar_width, label=feature, color=modified_cmap(i / len(features)))

ax1.set_xlabel('Minority ratio')
ax1.set_ylabel('Macro average F1 score (%)')
ax1.set_xticks(index + bar_width * (len(features) - 1) / 2)
ax1.set_xticklabels(minority_ratios)
ax1.legend()
ax1.set_ylim(bottom=45, top=75)

# Plot Minority F1
for i, feature in enumerate(features):
    ax2.bar(index + i * bar_width, minority_f1[:, i], width=bar_width, label=feature, color=modified_cmap(i / len(features)))

ax2.set_xlabel('Minority ratio')
ax2.set_ylabel('Minority class F1 score (%)')
ax2.set_xticks(index + bar_width * (len(features) - 1) / 2)
ax2.set_xticklabels(minority_ratios)
ax2.legend()
ax2.set_ylim(bottom=0, top=65)

plt.tight_layout()
plt.savefig('./data/' + 'ImbalancedF1Comparison', dpi=600)
plt.show()
