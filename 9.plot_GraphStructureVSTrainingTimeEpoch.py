import matplotlib.pyplot as plt
import numpy as np

ACHG_structure = ['ACHG-All', 'ACHG-M', 'ACHG-D', 'ACHG-T', 'ACHG-N']
Embedding_time = [5607.931312, 3197.64388, 3326.878481, 4410.253901, 3159.647718]
Embedding_epoch = [121, 100, 105, 105, 108]

# Create subplots
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(9, 3.5))

# Bar chart for subplot a (Embedding Time)
ax_a.bar(ACHG_structure, Embedding_time, color='#1f77b4', label='Embedding time')
ax_a.set_xlabel('ACHG structure')
ax_a.set_ylabel('Embedding time (s)')
ax_a.set_ylim(bottom=3000, top=max(Embedding_time) + 500)  # Adjust the top limit as needed

# Annotate each bar in subplot a
for i, value in enumerate(Embedding_time):
    ax_a.text(i, value + 100, f'{value:.2f}', ha='center', va='bottom')

ax_a.legend()

# Bar chart for subplot b (Training Epoch)
ax_b.bar(ACHG_structure, Embedding_epoch, color='#ff7f0e', label='Embedding epoch')
ax_b.set_xlabel('ACHG structure')
ax_b.set_ylabel('Embedding epoch')
ax_b.set_ylim(bottom=90, top=125)  # Adjust the top limit as needed

# Annotate each bar in subplot b
for i, value in enumerate(Embedding_epoch):
    ax_b.text(i, value + 1, str(value), ha='center', va='bottom')

ax_b.legend()

# Adjust layout and show the plots
plt.tight_layout()
plt.savefig('./data/' + 'GraphStructureVSTrainingTimeEpoch', dpi=600)
plt.show()
