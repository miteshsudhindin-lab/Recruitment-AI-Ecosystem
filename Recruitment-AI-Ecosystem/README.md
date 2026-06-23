# AI Recruitment Ecosystem & ATS Pro 🚀

An end-to-end, locally deployed AI-driven Applicant Tracking System (ATS) and Recruitment Ecosystem. This project automates the ingestion of job descriptions, processes candidate resumes, and intelligently routes profiles using Machine Learning.

## 🧰 Tech Stack & Architecture

- **Data Ingestion:** Web scraping pipeline built with `requests` and `BeautifulSoup` to dynamically compile active roles into `scraped_jobs.csv`.
- **NLP Pipeline:** Custom text normalization and cleaning using `NLTK` / `spaCy`, coupled with TF-IDF Vectorization.
- **Machine Learning Brain:** A trained **Multinomial Naive Bayes** classifier achieving **95% accuracy** in domain routing (Data Science/AI vs. HR). Models are serialized and loaded via `pickle`.
- **Parsing Engine:** Robust PDF parsing utilizing `pypdf` with optimized Regex patterns for real-time contact metadata extraction (Emails, Phone numbers).
- **User Interface:** Interactive UI dashboard built completely with `Streamlit` featuring real-time file upload parsing, progress bars, and structured match breakdowns.

## 📁 Repository Structure

- `app.py` - The core Streamlit dashboard application.
- `nb_resume_model.pkl` - Serialized Naive Bayes model.
- `tfidf_vectorizer.pkl` - Serialized text vectorizer.
- `scraped_jobs.csv` - Scraped job dataset.

## 🚀 How to Run Locally

1. Clone the repository:
   ```bash
   git clone [https://github.com/miteshsudhindin-lab/Recruitment-AI-Ecosystem.git](https://github.com/miteshsudhindin-lab/Recruitment-AI-Ecosystem.git)
   cd Recruitment-AI-Ecosystem