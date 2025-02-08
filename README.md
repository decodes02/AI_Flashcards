# 📚 AI-Powered Flashcard Learning System

An **AI-powered flashcard web application** that helps users **create, review, analyze**, and get **AI-based suggestions** for flashcards. The system is designed to improve learning efficiency using **TF-IDF similarity-based recommendations**.

---

## 🌟 Features

✅ **Create Flashcards** with question and answer.  
✅ **Review Flashcards** with next/previous navigation.  
✅ **Mark Flashcards as Reviewed** and track review count.  
✅ **View Review Analytics** to see frequently reviewed cards.  
✅ **AI-Based Suggestions** to get related flashcards using **TF-IDF & Cosine Similarity**.  
✅ **Responsive UI** for a seamless experience across devices.  

---

## 🛠 Tech Stack

| Component  | Technology Used  |
|------------|----------------|
| **Frontend** | HTML, CSS (Bootstrap), JavaScript (Fetch API) |
| **Backend**  | Flask (Python) |
| **Database**  | MySQL |
| **AI Model** | TF-IDF Vectorizer + Cosine Similarity (Scikit-Learn) |
| **Deployment** | Flask Server |

---

## 🚀 Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/flashcard-ai.git
cd flashcard-ai
```
## 🖥️ API Endpoints

| **Method** | **Endpoint**       | **Description**                        |
|-----------|------------------|------------------------------------|
| `POST`    | `/flashcards`      | Add new flashcards               |
| `GET`     | `/review`          | Get all flashcards               |
| `POST`    | `/review/<id>`     | Mark a flashcard as reviewed     |
| `GET`     | `/analytics`       | Get review analytics             |
| `GET`     | `/ai-suggestions`  | Get AI-based flashcard suggestions |


## 🎨 User Interface

The frontend uses **Bootstrap** for a responsive UI and includes:

- 📖 **Flashcard navigation** (Next/Previous)
- 📊 **Analytics section** to track progress
- 🤖 **AI-based suggestions** for related flashcards

---

## 🔥 Future Improvements

- 📌 **User authentication** (Login & progress tracking)
- 🎯 **Adaptive difficulty adjustment** based on performance
- 📊 **More detailed analytics** on learning progress

---

## 🤝 Contributing

Contributions are welcome! If you'd like to improve the project, feel free to:

1. 🍴 Fork the repo  
2. 🛠️ Create a new branch:  
   ```bash
   git checkout -b feature-name
