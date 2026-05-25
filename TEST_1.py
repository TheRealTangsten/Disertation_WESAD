import neurokit2 as nk
import matplotlib.pyplot as plt
import seaborn as sns
# Download example data
#data = nk.data("bio_eventrelated_100hz")
import data_loading as dataLoading
from sklearn.preprocessing import LabelEncoder

# Preprocess the data (filter, find peaks, etc.)
#processed_data, info = nk.bio_process(ecg=data["ECG"], rsp=data["RSP"], eda=data["EDA"], sampling_rate=100)

# Compute relevant features
#results = nk.bio_analyze(processed_data, sampling_rate=100)
full_df = dataLoading.load_processed_data(json_type="chest", include_resp=True)
print(full_df)
print(full_df[full_df.Label==0].shape[0])
print(full_df[full_df.Label==1].shape[0])
print(full_df[full_df.Label==2].shape[0])
print(set(full_df['Label']))




le = LabelEncoder()
full_df['Label'] = le.fit_transform(full_df['Label'])
print(full_df)
print(full_df[full_df.Label==0].shape[0])
print(full_df[full_df.Label==1].shape[0])
print(full_df[full_df.Label==2].shape[0])
print(set(full_df['Label']))


list1 =[1,2,3]
list2 =[4,5,6]
list3 =[7,8,9]
list4 = list(list1,list2,list3)
print(list4)

"""
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
"""