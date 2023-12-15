# Handwritten Digit Recognition 
Project for the course CS319 Computer Vision, American University of Armenia
Professor: Arman Chopikyan
Contributors: 
- Anna Charchyan
- Zaruhi Poghosyan
- Naira Maria Barseghyan

## Project Idea
This project is designed to create a model for hand written digit recognition. We created a Convolutional Neural Network (CNN) that trains on a MNIST hand wirtten digit data and makes predictions. The CNN is implemented using PyTorch.


## Code Structure
project/
    |-- Code/
    |   |-- dataset.py
    |   |-- model.py
    |   |-- train.py
    |   |-- inference.py
    |-- Data/
    |   |-- train/
    |   |-- validation/
    |   |-- test/
    |-- README.md
    |-- requirements.txt

## Installation Guide

Install the code to your local environement.
```bash
git clone ---
```

It is recomended to use virtual environment with python3.9.
```bash
python3.9 -m venv "venv"
```
Activate environment.
```bash
venv/bin/activate 
```

Install requirements.
```bash
pip install -r requirements.txt
```

## Uasge Guide
To train the model run train.py. Model trains on mnist dataset.

```bash
python train.py
```

After training the model run the inference.py to test the model on data in test folder.

```bash
python inference.py
```
