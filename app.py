import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import google.generativeai as genai

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="RAG Analisis Kepuasan Pelanggan PRDECT-ID",
    page_icon="🛍️",
    layout="wide"
)

# --- PENANGANAN GEMINI API KEY ---
api_key = None
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
elif "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]

with st.sidebar:
    st.header("⚙️ Pengaturan RAG")
    if not api_key:
        api_key = st.text_input(
            "Masukkan Gemini API Key:", 
            type="password", 
            help="Dapatkan API Key gratis di https://aistudio.google.com/apikey"
        )
    
    top_k = st.slider(
        "Jumlah dokumen yang diambil (Top-K):",
        min_value=3,
        max_value=10,
        value=5
    )

# --- LOAD DATASET ---
@st.cache_data
def load_data():
    df = pd.read_csv("PRDECT-ID_Dataset.csv")
    return df

try:
    df = load_data()
except Exception as e:
    st.error("File 'PRDECT-ID_Dataset.csv' tidak ditemukan. Pastikan file tersimpan di root direktori repository GitHub Anda.")
    st.stop()

# --- KAMUS NORMALISASI SLANG & CLEANING TEKS ---
slang_dict = {
    "gk": "tidak", "ga": "tidak", "gak": "tidak", "tdk": "tidak", "bgs": "bagus",
    "trimakasih": "terima kasih", "trims": "terima kasih", "recomend": "rekomendasi",
    "recommended": "rekomendasi", "oke": "ok", "okee": "ok", "mantul": "mantap betul",
    "cpt": "cepat", "bnyk": "banyak", "dtg": "datang", "sdh": "sudah", "udah": "sudah",
    "udh": "sudah", "pesenan": "pesanan"
}

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [slang_dict.get(tok, tok) for tok in tokens]
    return " ".join(tokens)

df["clean_text"] = df["Customer Review"].apply(clean_text)

# --- BUILD INDEX TF-IDF ---
@st.cache_resource
def build_index():
    vectorizer = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2)
    )
    corpus = df["clean_text"] + " " + df["Category"].astype(str).str.lower()
    matrix = vectorizer.fit_transform(corpus)
    return vectorizer, matrix

vectorizer, tfidf_matrix = build_index()

# --- FUNGSI RETRIEVAL ---
def retrieve(query, top_k=5):
    clean_q = clean_text(query)
    query_vector = vectorizer.transform([clean_q])
    similarity = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_index = similarity.argsort()[::-1][:top_k]

    documents = []
    for idx in top_index:
        documents.append({
            "doc_id": int(idx),
            "score": float(similarity[idx]),
            "category": df.iloc[idx]["Category"],
            "product": df.iloc[idx]["Product Name"],
            "review": df.iloc[idx]["Customer Review"],
            "sentiment": df.iloc[idx]["Sentiment"]
        })
    return documents

# --- FUNGSI GENERATE ANSWER (GOOGLE GEMINI - AUTO DETECT MODEL) ---
def generate_answer(question, docs, user_api_key):
    genai.configure(api_key=user_api_key)
    
    # 1. Cari model yang didukung secara otomatis dari API Key Anda
    target_model_name = None
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                # Prioritaskan model flash jika ada
                if 'flash' in m.name:
                    target_model_name = m.name
                    break
                elif not target_model_name:
                    target_model_name = m.name
    except Exception as e:
        st.warning(f"Gagal melakukan auto-detect model: {e}")

    # Fallback jika list_models gagal
    if not target_model_name:
        target_model_name = "gemini-1.5-flash"

    # 2. Inisialisasi Model
    model = genai.GenerativeModel(target_model_name)
    
    context = ""
    for i, d in enumerate(docs):
        context += f"""
Dokumen {i+1}
Kategori : {d['category']}
Produk   : {d['product']}
Sentimen : {d['sentiment']}
Review   : {d['review']}
--------------------------------------
"""

    prompt = f"""
Anda adalah AI Assistant yang bertugas menganalisis kepuasan pelanggan e-commerce Indonesia berdasarkan dataset ulasan PRDECT-ID.

Jawablah pertanyaan pengguna HANYA berdasarkan konteks ulasan yang diberikan berikut ini.
Jangan menggunakan pengetahuan atau asumsi di luar ulasan ini.
Jika informasi pada dokumen ulasan tidak mencukupi, katakan secara jujur bahwa data ulasan belum mencukupi.

Aturan Jawaban:
1. Buat jawaban ringkas, terstruktur, dan analisis yang natural.
2. Rangkum temuan utama (poin positif/negatif) bukan sekadar menyalin teks ulasan.
3. Di bagian akhir, sebutkan secara jelas dokumen referensi mana saja (misal: Dokumen 1, Dokumen 3) yang mendukung jawaban Anda.

==============================
PERTANYAAN:
{question}

==============================
KONTEKS ULASAN:
{context}

==============================
JAWABAN:
"""

    response = model.generate_content(prompt)
    
    if response and hasattr(response, 'text') and response.text:
        return response.text
    elif response and response.candidates:
        return response.candidates[0].content.parts[0].text
    else:
        return "Maaf, AI tidak dapat menghasilkan jawaban untuk pertanyaan ini."

# --- INTERFACE UTAMA ---
st.title("🛍️ RAG - Analisis Kepuasan Pelanggan PRDECT-ID")
st.markdown("""
Prototipe **Retrieval-Augmented Generation (RAG)** untuk Analisis Kepuasan Pelanggan E-Commerce Indonesia.  
*Arsitektur: TF-IDF Retrieval + Cosine Similarity + Google Gemini LLM Generation*
""")

# Sidebar Info
with st.sidebar:
    st.markdown("---")
    st.subheader("📊 Informasi Dataset")
    st.write(f"Total Ulasan: **{len(df):,}**")
    st.write(f"Total Kategori: **{df['Category'].nunique()}**")
    pos_count = (df['Sentiment'].astype(str).str.capitalize() == 'Positive').sum()
    neg_count = (df['Sentiment'].astype(str).str.capitalize() == 'Negative').sum()
    st.write(f"Review Positif: **{pos_count:,}**")
    st.write(f"Review Negatif: **{neg_count:,}**")

    st.markdown("---")
    st.subheader("💡 Contoh Pertanyaan")
    st.caption("• Bagaimana kepuasan pelanggan pada kategori Beauty?")
    st.caption("• Apa keluhan pelanggan terhadap produk Electronics?")
    st.caption("• Mengapa pelanggan memberikan review negatif?")
    st.caption("• Bagaimana kualitas pengiriman dan kemasan produk?")

# Form Pertanyaan
question = st.text_area(
    "Masukkan Pertanyaan Analisis:",
    height=100,
    placeholder="Contoh: Bagaimana kepuasan pelanggan terhadap kualitas produk kecantikan?"
)

run = st.button("🔍 Cari Jawaban & Analisis", use_container_width=True)

if run:
    if not question.strip():
        st.warning("Silakan masukkan pertanyaan terlebih dahulu.")
    elif not api_key:
        st.error("Gemini API Key belum diisi. Masukkan API Key di sidebar atau tambahkan ke Streamlit Secrets.")
    else:
        with st.spinner("🔍 Melakukan retrieval dokumen relevan..."):
            docs = retrieve(question, top_k)

        with st.spinner("🤖 Menyusun analisis menggunakan Google Gemini..."):
            try:
                answer = generate_answer(question, docs, api_key)
                st.success("Analisis Berhasil Disusun!")
                st.subheader("💬 Hasil Analisis AI")
                st.markdown(answer)  # Menggunakan st.markdown agar format teks terstruktur rapi
            except Exception as e:
                st.error(f"Gagal menghasilkan jawaban dari Gemini API: {e}")

        st.markdown("---")
        st.subheader("📚 Dokumen Referensi (Hasil Retrieval)")
        for i, d in enumerate(docs):
            with st.expander(f"Dokumen {i+1} | {d['category']} | Score: {d['score']:.4f} | Sentimen: {d['sentiment']}"):
                st.write(f"**Produk:** {d['product']}")
                st.write(f"**Ulasan:** {d['review']}")

st.markdown("---")
st.caption("Universitas Islam Indonesia | Mata Kuliah Trending Topics on Statistics | Prototipe Retrieval-Augmented Generation")
