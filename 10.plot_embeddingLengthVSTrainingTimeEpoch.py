import matplotlib.pyplot as plt
import numpy as np

Embedding_length = [16, 32, 64, 128, 256]
Training_time = [7544.283504, 5949.250861, 3159.647718, 3683.682754, 2561.929137]
Training_epoch = [217, 149, 108, 87, 57]

# Create subplots
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(9, 3.5))

# Plot for subplot a
ax_a.plot(Embedding_length, Training_time, 'o-', color='#1f77b4', label='Embedding time')
ax_a.set_xlabel('Embedding length')
ax_a.set_ylabel('Embedding time (s)')

# Annotate each point in subplot a
for i, txt in enumerate(Training_time):
    ax_a.annotate(f'{txt:.2f}', (Embedding_length[i], Training_time[i]), textcoords="offset points", xytext=(0, 5), ha='center')

ax_a.legend()

# Set x-axis ticks for subplot a
ax_a.set_xticks(Embedding_length)
# Set y-axis limits for subplot a
ax_a.set_ylim(bottom=2000, top=8300)

# Set y-axis limits for subplot a
ax_a.set_xlim(left=-20,right=285)

# Plot for subplot b
ax_b.plot(Embedding_length, Training_epoch, 'o-', color='#ff7f0e',label='Embedding epoch')
ax_b.set_xlabel('Embedding length')
ax_b.set_ylabel('Embedding epoch')

# Annotate each point in subplot b
for i, txt in enumerate(Training_epoch):
    ax_b.annotate(f'{txt}', (Embedding_length[i], Training_epoch[i]), textcoords="offset points", xytext=(0, 5), ha='center')

ax_b.legend()

# Set x-axis ticks for subplot b
ax_b.set_xticks(Embedding_length)

# Set y-axis limits for subplot b
ax_b.set_ylim(bottom=50, top=250)

# Set y-axis limits for subplot a
ax_a.set_xlim(left=-10)

# Adjust layout and show the plots
plt.tight_layout()
plt.savefig('./data/' + 'EmbeddingLengthVSTrainingTimeEpoch', dpi=600)
plt.show()
