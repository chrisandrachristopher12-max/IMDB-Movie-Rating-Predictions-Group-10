# 🎬 IMDB Movie Rating Prediction
![KD34403](https://img.shields.io/badge/KD34403-Machine%20Learning-blue) ![UMS](https://img.shields.io/badge/Universiti%20Malaysia%20Sabah-Group%2010-green)

## Project overview
This project develops a machine learning model to predict IMDb movie ratings using movie features such as genre, runtime, release year, number of votes, and gross revenue. The data is preprocessed using scaling and one-hot encoding, and a Gradient Boosting Regressor is trained to estimate ratings on a scale of 1 to 10. The goal is to build an accurate model that can predict ratings for new movies based on their metadata.

## Repository structure
```
IMDB-Movie-Rating-Predictions-Group-10/
├── README.md
├── requirements.txt
├── data/
│   └── README.md          ← dataset download link here
├── notebooks/
│   ├── 01_Data_Pipeline.ipynb
│   ├── 02_Architecture_Logic.ipynb
│   ├── 03_Training_Loop.ipynb
│   ├── 04_Model_Optimization.ipynb
│   └── 05_Final_Evaluation.ipynb
└── models/
    └── .gitkeep
```

## Dataset
- **Source:** [IMDB Top 1000 Movies – Kaggle](https://www.kaggle.com/datasets/thedevastator/imdb-movie-ratings-dataset)
- **Features:** Genre, Runtime, IMDB Rating, No. of Votes, Gross Revenue, Director, Release Year
- **Target variable:** `IMDB_Rating` (continuous, 1.0 – 10.0)

## Milestone videos
| Milestone | Topic | YouTube Link |
|-----------|-------|--------------|
| M1 | Data Pipeline | [1](https://youtu.be/NVMA_dkZF3Q) |
| M2 | Architecture Logic | [2](https://youtu.be/RrCvqTxPDC8) |
| M3 | Training Loop | [3](https://youtu.be/nr_Gt__fmdY)|
| M4 | Model Optimization | [4](https://youtu.be/f8io57OuWWw?si=Ti8FtYU5JyUYlx_c) |
| M5 | Final Evaluation | [5](https://youtu.be/7jstQpHVtwA) |

## How to run
```bash
pip install -r requirements.txt
jupyter notebook
```
Open the notebooks in order (01 → 05). Each notebook is self-contained for its milestone.

## 👥 Group members
| Name | GitHub username | Milestone |
|------|----------------|-----------|
| FARHANI SYAMSUDDIN BI23110050 | farhanisyamsuddin12 | M1 – Data Pipeline |
| JULAIKA ANG BI23110160 | jul29-bit | M2 – Architecture Logic |
| CHRISANDRA BUSAK CHRISTOPHER BI23110160 | chrisandrachirstopher12-max | M3 – Training Loop |
| NORAIN AQILAH BINTI SHAMSUDDIN BI22110478 | aiinaqilah | M4 – Model Optimization |
| NABILA GOTIMUS BI23110288 | assyahidNabila | M5 – Final Evaluation |
