import neurokit2 as nk
import matplotlib.pyplot as plt
import seaborn as sns
# Download example data
#data = nk.data("bio_eventrelated_100hz")

# Preprocess the data (filter, find peaks, etc.)
#processed_data, info = nk.bio_process(ecg=data["ECG"], rsp=data["RSP"], eda=data["EDA"], sampling_rate=100)

# Compute relevant features
#results = nk.bio_analyze(processed_data, sampling_rate=100)

test_matrix = [ [1,2,3], [4,5,6], [7,8,9] ]
classes = ['Baseline', 'Relaxed', 'Stress']


fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle(f'Test', fontsize=16)

# model_names = ['Random Forest', 'CNN', 'Transformer']
model_names = ['Random Forest', 'CNN', 'Transformer', 'LSTM']
predictions = [test_matrix, test_matrix, test_matrix, test_matrix]

axes = axes.flatten()
print(axes)
for i, ax in enumerate(axes):
    print(i, ax)
for i, ax in enumerate(axes):

    sns.heatmap(test_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                xticklabels=classes, yticklabels=classes)

    ax.set_title(f'{model_names[i]}')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

plt.tight_layout()
plt.show()