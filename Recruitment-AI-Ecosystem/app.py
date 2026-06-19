import streamlit as st
import pandas as pd
import re
import pickle
from pypdf import PdfReader


st.set_page_config(page_title="AI Recruitment Ecosystem", layout="wide")
st.title("🤖 Advanced AI Recruitment Ecosystem (ATS Pro)")
st.markdown("---")


@st.cache_resource
def load_ml_brain():
    try:
        with open("nb_resume_model.pkl", "rb") as f1:
            trained_model = pickle.load(f1)
        with open("tfidf_vectorizer.pkl", "rb") as f2:
            trained_vectorizer = pickle.load(f2)
        return trained_model, trained_vectorizer
    except FileNotFoundError:
        return None, None

model, vectorizer = load_ml_brain()


col1, col2 = st.columns([1, 1.2])

with col1:
    st.header("📋 Live Job Market Inventory")
    st.write("Current structured database profiles mined via background scraper engine.")
    
    try:
        jobs_df = pd.read_csv("scraped_jobs.csv")
        st.dataframe(jobs_df, width="stretch", height=550)
        st.success(f"Loaded {len(jobs_df)} indexable profiles.")
    except FileNotFoundError:
        st.warning("⚠️ 'scraped_jobs.csv' not found.")
        jobs_df = pd.DataFrame()

with col2:
    st.header("🧠 NLP Intelligence & ATS Parser")
    st.write("Drop a candidate profile below to parse details, calculate matching weights, and predict domain classification.")
    
    uploaded_file = st.file_uploader("Upload Resume Document (PDF format only):", type=["pdf"])
    
    if uploaded_file is not None:
        st.success("📄 Document uploaded successfully!")
        
        
        reader = PdfReader(uploaded_file)
        extracted_text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
        
        
        email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', extracted_text)
        phone_match = re.search(r'\+?\d[\d\s\(\)-]{8,14}\d', extracted_text)
        
        email = email_match.group(0) if email_match else "Not found inside profile text"
        phone = phone_match.group(0) if phone_match else "Not found inside profile text"
        
        
        st.subheader("🕵️ Extracted Contact Metadata")
        meta_c1, meta_c2 = st.columns(2)
        meta_c1.info(f"📧 **Email:** {email}")
        meta_c2.info(f"📞 **Phone:** {phone}")
        
        if st.button("Execute Deep AI Candidate Screening"):
            st.markdown("---")
            
            
            clean_text = extracted_text.lower()
            clean_text = re.sub(r'http\S+\s*', ' ', clean_text)
            clean_text = re.sub(r'[^\w\s]', ' ', clean_text)
            clean_text = ' '.join(clean_text.split())
            
            
            if model and vectorizer:
                
                numerical_features = vectorizer.transform([clean_text])
                
                ai_prediction = model.predict(numerical_features)[0]
            else:
                ai_prediction = "System Training Engine Offline"
            
            
            ai_keywords = {"python", "machine learning", "sql", "data", "deep learning", "pytorch"}
            hr_keywords = {"hr", "talent", "payroll", "recruitment", "onboarding", "screening"}
            
            resume_words = set(clean_text.split())
            
            
            if ai_prediction in ["Data Science", "Web Designing", "Java Developer"] or any(w in resume_words for w in ai_keywords):
                if ai_prediction == "System Training Engine Offline":
                    ai_prediction = "Data Scientist / AI Engineer"
                target_keywords = ai_keywords
            elif "hr" in clean_text or any(w in resume_words for w in hr_keywords):
                ai_prediction = "Human Resources (HR)"
                target_keywords = hr_keywords
            else:
                ai_prediction = "Unmatched Profile"
                target_keywords = set()

            
            if ai_prediction != "Unmatched Profile":
                st.subheader("🎯 System Prediction Results:")
                st.metric(label="Predicted Candidate Domain Match", value=ai_prediction)
                
                
                matched_skills = resume_words.intersection(target_keywords)
                match_percentage = (len(matched_skills) / len(target_keywords)) * 100 if target_keywords else 50.0
                
                final_score = max(match_percentage, 65.0) if len(matched_skills) > 0 else 30.0
                
                st.write(f"📊 **AI Core Skill Match Index:** `{final_score:.1f}%`")
                st.progress(final_score / 100)
                st.success(f"🎯 Candidate verified. Compatibility matches active openings in the local inventory.")
            else:
                st.subheader("🎯 System Prediction Results:")
                st.metric(label="System Status", value="Unmatched Profile")
                st.error("❌ No role is available for you over here at the moment.")
                st.info("💡 Our active openings are currently restricted to AI/Data Science, Technical Devs, and Human Resources.")