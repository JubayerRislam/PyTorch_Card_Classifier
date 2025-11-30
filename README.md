# PyTorch Card Classifier 🃏

A deep learning project that classifies **53 playing cards** using **EfficientNet-B0** and **PyTorch**. Includes the training pipeline, pre-trained model, and a **Streamlit app** for real-time predictions.

---

## Features

- Classifies 53 card types (Hearts, Diamonds, Clubs, Spades)
- Uses **EfficientNet-B0** for feature extraction with transfer learning
- Preprocessing and data augmentation included
- Real-time **Streamlit app** for image upload and top-5 predictions
- Visualizes predictions with probabilities

---

## Dataset

The dataset is organized in **ImageFolder** format:

train/
├─ Ace of Hearts/
├─ 2 of Hearts/
├─ 3 of Hearts/
├─ ...
├─ King of Spades/

valid/
├─ Ace of Hearts/
├─ 2 of Hearts/
├─ 3 of Hearts/
├─ ...
├─ King of Spades/

test/
├─ Ace of Hearts/
├─ 2 of Hearts/
├─ 3 of Hearts/
├─ ...
├─ King of Spades/



- Each folder corresponds to a single card class
- 53 classes total

You can download the dataset from Kaggle:  
[Kaggle Card Dataset](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification)

---

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
