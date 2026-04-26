
import streamlit as st
import pandas as pd
import numpy as np
import faiss
import pickle
import os
import json
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import base64
from io import BytesIO
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import fitz
import docx
import re
from dataclasses import dataclass, asdict
import plotly.express as px
import io

# ================= CONFIG =================
DB_FILEPATH = "employment_platform_db.pkl"

# 🔥 FAST + LIGHT (important for your laptop)
TEXT_ENCODER_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

FAISS_DIMENSION = 384  # changed for MiniLM
MAX_LLM_NEW_TOKENS = 100


# ================= DATA MODEL =================
@dataclass
class CandidateProfile:
    id: str
    name: str
    email: str
    skills: List[str]
    total_experience_years: int
    experience_details: List[Dict]
    education: str
    job_titles: List[str]
    resume_text: str
    original_resume: Dict
    portfolio_files: List[Dict]
    text_embedding: np.ndarray
    upload_date: str


# ================= AI MODELS =================
@st.cache_resource
def get_cached_models():
    with st.spinner("Loading AI models..."):
        text_encoder = SentenceTransformer(TEXT_ENCODER_MODEL)
        llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
        llm_model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL
        )
        return text_encoder, llm_tokenizer, llm_model

class AIModels:
    def __init__(self):
        self.models_loaded = False
        self.text_encoder = None
        self.llm_model = None
        self.llm_tokenizer = None

    def load_models(self):
        if self.models_loaded:
            return
            
        try:
            self.text_encoder, self.llm_tokenizer, self.llm_model = get_cached_models()
            self.models_loaded = True
            st.success("AI Loaded!")
        except Exception as e:
            st.error(f"Model load failed: {e}")
            st.stop()


# ================= RESUME PARSER =================
class ResumeParser:
    def __init__(self, ai_models: AIModels):
        self.ai_models = ai_models

    def parse_pdf(self, file_bytes):
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        return "".join(page.get_text() for page in doc)

    def parse_docx(self, file_bytes):
        doc = docx.Document(BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs)

    def parse_with_llm(self, text):
        # Truncate text aggressively to speed up context prefill time
        short_text = text[:1500]
        
        prompt = f"""
Extract the top 5 technical skills from this resume.
Return ONLY valid JSON format exactly like this:
{{"skills": ["Skill1", "Skill2"]}}

Resume:
{short_text}
"""

        tokenizer = self.ai_models.llm_tokenizer
        model = self.ai_models.llm_model

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_LLM_NEW_TOKENS,
                temperature=0.3,
                do_sample=True
            )

        result = tokenizer.decode(outputs[0], skip_special_tokens=True)

        match = re.search(r"\{.*\}", result, re.DOTALL)
        if not match:
            return None

        try:
            return json.loads(match.group(0))
        except:
            return None


# ================= VECTOR DB =================
class VectorDatabase:
    def __init__(self):
        self.index = faiss.IndexFlatIP(FAISS_DIMENSION)
        self.candidates = {}
        self.db_filepath = DB_FILEPATH
        self.faiss_filepath = "employment_platform_faiss.index"
        self._load_db()

    def _load_db(self):
        if os.path.exists(self.db_filepath) and os.path.exists(self.faiss_filepath):
            try:
                with open(self.db_filepath, "rb") as f:
                    self.candidates = pickle.load(f)
                self.index = faiss.read_index(self.faiss_filepath)
            except Exception as e:
                print(f"Error loading DB: {e}")

    def _save_db(self):
        try:
            with open(self.db_filepath, "wb") as f:
                pickle.dump(self.candidates, f)
            faiss.write_index(self.index, self.faiss_filepath)
        except Exception as e:
            print(f"Error saving DB: {e}")

    def add_candidate(self, candidate):
        vec = candidate.text_embedding.astype('float32').reshape(1, -1)
        faiss.normalize_L2(vec)
        self.index.add(vec)
        self.candidates[self.index.ntotal - 1] = candidate
        self._save_db()

    def search(self, query_vec, k=5):
        query_vec = query_vec.astype('float32').reshape(1, -1)
        faiss.normalize_L2(query_vec)

        scores, indices = self.index.search(query_vec, k)

        return [
            (self.candidates[i], float(s))
            for s, i in zip(scores[0], indices[0])
            if i in self.candidates
        ]


# ================= MAIN PLATFORM =================
class Platform:
    def __init__(self):
        self.ai = AIModels()
        self.ai.load_models()
        self.db = VectorDatabase()
        self.parser = ResumeParser(self.ai)

    def process(self, name, email, file):
        file_bytes = file.getvalue()

        if file.name.lower().endswith(".pdf"):
            text = self.parser.parse_pdf(file_bytes)
        else:
            text = self.parser.parse_docx(file_bytes)

        parsed = self.parser.parse_with_llm(text)

        skills = parsed.get("skills", []) if parsed else []

        emb = self.ai.text_encoder.encode(text)

        candidate = CandidateProfile(
            id=str(len(self.db.candidates)+1),
            name=name,
            email=email,
            skills=skills,
            total_experience_years=0,
            experience_details=[],
            education="",
            job_titles=[],
            resume_text=text,
            original_resume={},
            portfolio_files=[],
            text_embedding=emb,
            upload_date=datetime.now().isoformat()
        )

        self.db.add_candidate(candidate)
        return candidate

    def search(self, job_desc):
        emb = self.ai.text_encoder.encode(job_desc)
        return self.db.search(emb)


# ================= UI =================
def main():
    st.set_page_config("LinkedOut", layout="wide")

    if "app" not in st.session_state:
        st.session_state.app = Platform()

    app = st.session_state.app

    st.title("🎯 LinkedOut AI Hiring")

    tab1, tab2 = st.tabs(["Candidate", "Employer"])

    with tab1:
        st.header("Upload Resume")

        name = st.text_input("Name")
        email = st.text_input("Email")
        file = st.file_uploader("Resume", type=["pdf", "docx"])

        submit_placeholder = st.empty()

        if submit_placeholder.button("Submit", key="submit_btn"):
            if name and email and file:
                # Swap button to Submitting and disable it
                submit_placeholder.button("Submitting...", disabled=True, key="submit_wait")
                
                with st.spinner("Analyzing Resume with AI (this may take up to a minute)..."):
                    c = app.process(name, email, file)
                
                # Swap button to Submitted
                submit_placeholder.button("Submitted!", disabled=True, key="submit_done")
                st.success(f"{c.name} successfully added to database!")
            else:
                st.warning("Please fill out all fields and upload a resume before submitting.")

    with tab2:
        st.header("Search Candidates")

        jd = st.text_area("Job Description")

        if st.button("Search"):
            results = app.search(jd)

            for c, score in results:
                st.subheader(c.name)
                st.write("Skills:", ", ".join(c.skills))
                st.write("Score:", score)


if __name__ == "__main__":
    main()
