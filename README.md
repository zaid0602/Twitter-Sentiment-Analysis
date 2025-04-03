# Twitter-Sentiment-Analysis

## 📌 Project Description
This project performs **Sentiment Analysis on Tweets** using **Machine Learning (SVM)**. The goal is to classify tweets as **positive** and **negative** based on their content. The dataset used contains labeled tweets that help train the model.

## 🛠️ Features
- Preprocessing of Twitter data (removal of URLs, mentions, and special characters).
- Feature extraction using TF-IDF or CountVectorizer.
- Training and evaluation using **Logistic Regression**.
- Visualization of sentiment distribution.

## 📂 Project Structure
```
📁 Twitter-Sentiment-Analysis
│── 📄 README.md
│── 📂 dataset/                # Folder for storing datasets (not included in repo)
│── 📂 notebooks/              # Jupyter notebooks for analysis & training
│── 📂 models/                 # Trained models (saved as .pkl)
│── 📄 requirements.txt        # Python dependencies
│── 📄 twitter_sentiment_analysis.ipynb  # Main notebook
```

## 🚀 How to Run This Project

### 1️⃣ Clone the Repository
```sh
git clone https://github.com/zaid0602/Twitter-Sentiment-Analysis.git
cd Twitter-Sentiment-Analysis
```

### 2️⃣ Create a Virtual Environment (Optional but Recommended)
```sh
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

### 4️⃣ Download Dataset
The dataset is **too large to be included in GitHub**. Download it manually from [Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140) and place it inside the `dataset/` folder.

### 5️⃣ Run the Jupyter Notebook
```sh
jupyter notebook
```
Open `twitter_sentiment_analysis.ipynb` and run the cells to train and test the model.

### 6️⃣ (Optional) Run as a Script
To test the model with a custom tweet:
```sh
python predict.py "I love this product! It's amazing."
```

## 📊 Results & Accuracy
The model is trained using **SVM**, achieving an accuracy of around **85%** on test data. Additional improvements can be made using **Deep Learning models like LSTMs**.

## 📝 Future Improvements
- Implement **Deep Learning-based Sentiment Analysis**.
- Deploy the model using a **Flask/Django API**.
- Create a simple **frontend for user input**.

## 🤝 Contributing
Feel free to fork this repository and submit **pull requests**! If you encounter any issues, open an **issue** in this repo.

---
📧 **Author**: Zaid Mustafa  
🔗 [GitHub](https://github.com/zaid0602)

