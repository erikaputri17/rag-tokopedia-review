import streamlit as st
import pandas as pd
import numpy as np
import re
import string
from mistralai import Mistral
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="RAG Analisis Kepuasan Pelanggan PRDECT-ID",
    page_icon="🛍️",
    layout="wide"
)

# --- PENANGANAN MISTRAL API KEY ---
api_key = None
if "MISTRAL_API_KEY" in st.secrets:
    api_key = st.secrets["MISTRAL_API_KEY"]

with st.sidebar:
    st.header("⚙️ Pengaturan RAG")
    if not api_key:
        api_key = st.text_input(
            "Masukkan Mistral API Key:", 
            type="password", 
            help="Dapatkan API Key gratis di https://console.mistral.ai"
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

# --- FUNGSI GENERATE ANSWER (MISTRAL AI) ---
def generate_answer(question, docs, user_api_key):
    client = Mistral(api_key=user_api_key)
    
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

    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# --- INTERFACE UTAMA ---
st.title("🛍️ RAG - Analisis Kepuasan Pelanggan PRDECT-ID")
st.markdown("""
Prototipe **Retrieval-Augmented Generation (RAG)** untuk Analisis Kepuasan Pelanggan E-Commerce Indonesia.  
*Arsitektur: TF-IDF Retrieval + Cosine Similarity + Mistral AI LLM Generation*
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
        st.error("Mistral API Key belum diisi. Masukkan API Key di sidebar atau tambahkan ke Streamlit Secrets.")
    else:
        with st.spinner("🔍 Melakukan retrieval dokumen relevan..."):
            docs = retrieve(question, top_k)

        with st.spinner("🤖 Menyusun analisis menggunakan Mistral AI..."):
            try:
                answer = generate_answer(question, docs, api_key)
                st.success("Analisis Berhasil Disusun!")
                st.subheader("💬 Hasil Analisis AI")
                st.write(answer)
            except Exception as e:
                st.error(f"Gagal menghasilkan jawaban dari Mistral API: {e}")

        st.markdown("---")
        st.subheader("📚 Dokumen Referensi (Hasil Retrieval)")
        for i, d in enumerate(docs):
            with st.expander(f"Dokumen {i+1} | {d['category']} | Score: {d['score']:.4f} | Sentimen: {d['sentiment']}"):
                st.write(f"**Produk:** {d['product']}")
                st.write(f"**Ulasan:** {d['review']}")

st.markdown("---")
st.caption("Universitas Islam Indonesia | Mata Kuliah Trending Topics on Statistics | Prototipe Retrieval-Augmented Generation")
