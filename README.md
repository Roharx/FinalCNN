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

---

## Output

Output will look like this:
```
cat
```