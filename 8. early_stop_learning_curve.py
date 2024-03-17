
import matplotlib.pyplot as plt
import numpy as np
import csv

# save result:
csv_file_path = "./data/Ablation_study_training_progress.csv"
# train_loss_list = []
# validation_loss_list=[]

with open(csv_file_path, mode='r') as file:
    csv_reader = csv.reader(file)
    # Read the first five lines

    setting1 = next(csv_reader)  # Replace 0 with the index of the column you want
    setting2 = next(csv_reader)  # Replace 1 with the index of the column you want
    setting3 = next(csv_reader)  # Replace 2 with the index of the column you want
    train_loss_list = next(csv_reader)   # Replace 3 with the index of the column you want
    validation_loss_list = next(csv_reader)   # Replace 4 with the index of the column you want


# early_stopping.draw_trend(train_loss_list, validation_loss_list)
train_list = [float(item) for item in train_loss_list]
test_list = [float(item) for item in validation_loss_list]

metric = 'loss'

plt.figure(figsize=(5, 3))
# Set the font size for axis labels and title
plt.rc('axes', labelsize=9)
plt.rc('axes', titlesize=9)
#
plt.plot(list(range(7, len(train_list) + 1)), train_list[6:], label='Training ' + metric)
plt.plot(list(range(7, len(test_list) + 1)), test_list[6:], label='Validation ' + metric)

# plt.plot(train_list, label='Training ' + metric)
# plt.plot(test_list, label='Validation ' + metric)

checkpoint = test_list.index(min(test_list)) + 1

plt.axvline(checkpoint, linestyle='--', color='r', label='Early stop checkpoint')

plt.xlabel('epoch',fontsize=10)
plt.ylabel(metric,fontsize=10)
# plt.ylim(min(train_list + test_list), max(train_list + test_list))  # consistent scale
# plt.xlim(1, len(test_list) + 1)  # consistent scale
plt.grid(True)
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig('./data/' + 'Traning_Eearly_Stop', dpi=600)
plt.show()
