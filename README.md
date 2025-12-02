## Dogs & Cats Dataset

Used dataset:

Cat and Dog: https://www.kaggle.com/datasets/tongpython/cat-and-dog (10k images of varying
sizes).

---

## Requirements

Install all required dependencies inside your virtual environment:

```
pip install tensorflow scipy keras_preprocessing numpy
```

### Required Python packages
- TensorFlow
- SciPy (required for zoom, shear, and flip augmentations)
- keras_preprocessing
- NumPy

---

## Folder Structure

Folder layout should look like this:

```
pythonProject/
│
├── CNN.py
├── predict.py
├── cat_dog_model.py
├── README.md
│
├── training_set/
│   ├── cats/
│   └── dogs/
│
├── test_set/
│   ├── cats/
│   └── dogs/
│
└── single_prediction/
    └── prediction.png
```

Make sure if you use the predict.py after training to adjust the name of the
file it reads out of to match the file name in the CNN.py.

The predict.py is not part of the submission, it is just there for my own convenience.
For this reason I will only update it to match my own file system that is excluded via .gitingore.

---

## Output

Output will look like this:
```
cat
```