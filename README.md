# 🎬 IMDB Movie Rating Prediction
![KD34403](https://img.shields.io/badge/KD34403-Machine%20Learning-blue) ![UMS](https://img.shields.io/badge/Universiti%20Malaysia%20Sabah-Group%2010-green)

## 📌 Project overview
This project builds a machine learning pipeline to predict IMDB movie ratings based on structured movie metadata such as genre, runtime, number of votes, release year, and gross revenue. The goal is to train a regression model that accurately estimates a movie's rating on a 1–10 scale.

## 📂 Repository structure
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

## 📊 Dataset
- **Source:** [IMDB Top 1000 Movies – Kaggle](https://www.kaggle.com/datasets/thedevastator/imdb-movie-ratings-dataset)
- **Features:** Genre, Runtime, IMDB Rating, No. of Votes, Gross Revenue, Director, Release Year
- **Target variable:** `IMDB_Rating` (continuous, 1.0 – 10.0)

## ▶️ Milestone videos
| Milestone | Topic | YouTube Link |
|-----------|-------|--------------|
| M1 | Data Pipeline | [1](https://youtu.be/NVMA_dkZF3Q) |
| M2 | Architecture Logic | [2](https://youtu.be/RrCvqTxPDC8) |
| M3 | Training Loop | [3](https://youtu.be/nr_Gt__fmdY)|
| M4 | Model Optimization | [4](https://youtu.be/f8io57OuWWw?si=Ti8FtYU5JyUYlx_c) |
| M5 | Final Evaluation | [5](https://youtu.be/7jstQpHVtwA) |

## ⚙️ How to run
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
| NABILA GOTIMUS | assyahidNabila | M5 – Final Evaluation |
