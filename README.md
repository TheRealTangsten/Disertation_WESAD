Link download set de date:
https://www.kaggle.com/datasets/orvile/wesad-wearable-stress-affect-detection-dataset/data

ATENTIE!
Numpy trebuie sa fie o versiune 1.26.x !!!


Pentru preprocesarea datelor brute si stocarea lor, inserati linia urmatoare de cod in main:
dataLoading.preprocess_and_save_to_json(ALL_SUBJECTS)

Modificati fisierul constants.py:
path_data = '<path-to-raw-data>'