"""
Streamlit RAG App - Analisis Kepuasan Pelanggan PRDECT-ID
Cara menjalankan:
  Lokal / Colab   : streamlit run app.py
  Streamlit Cloud : push repo ini ke GitHub (sertakan app.py, requirements.txt,
                    PRDECT-ID_Dataset.csv) lalu deploy di share.streamlit.io

API key LLM (Google Gemini, gratis) diminta lewat sidebar saat aplikasi dibuka,
atau bisa diisi lewat st.secrets["GOOGLE_API_KEY"] saat deploy ke Streamlit Cloud.
"""
import streamlit as st
import pandas as pd
import numpy as np
import re, string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="RAG - Kepuasan Pelanggan PRDECT-ID", page_icon="🛍️", layout="wide")

# ----------------------------------------------------------------------
# 1. LOAD & PREPARE DATA (cache agar tidak diproses ulang setiap interaksi)
# ----------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("PRDECT-ID_Dataset.csv")

    def clean(text):
        t = str(text).lower()
        t = re.sub(r'http\S+|www\S+', ' ', t)
        t = re.sub(r'[^\x00-\x7F]+', ' ', t)
        t = re.sub(r'\d+', ' ', t)
        t = t.translate(str.maketrans('', '', string.punctuation))
        return re.sub(r'\s+', ' ', t).strip()

    df['clean_text'] = df['Customer Review'].apply(clean)
    df['kb_text'] = (df['Category'] + ' - ' + df['Product Name'] + '. Ulasan: ' + df['Customer Review']).astype(str)
    return df

@st.cache_resource
def build_index(df):
    vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(df['clean_text'] + ' ' + df['Category'].str.lower())
    return vectorizer, matrix

df = load_data()
vectorizer, kb_matrix = build_index(df)

# ----------------------------------------------------------------------
# 2. RETRIEVAL
# ----------------------------------------------------------------------
def retrieve(query, top_k=5):
    q = re.sub(r'[^\x00-\x7F]+', ' ', query.lower())
    q_vec = vectorizer.transform([q])
    sims = cosine_similarity(q_vec, kb_matrix).ravel()
    idx = sims.argsort()[::-1][:top_k]
    return [{
        "doc_id": int(i), "score": float(sims[i]),
        "category": df.iloc[i]["Category"], "product": df.iloc[i]["Product Name"],
        "review": df.iloc[i]["Customer Review"], "sentiment": df.iloc[i]["Sentiment"]
    } for i in idx]

# ----------------------------------------------------------------------
# 3. GENERATION (Google Gemini)
# ----------------------------------------------------------------------
def generate_answer(api_key, question, docs):
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    context = "\n\n".join(
        f"[Dok {i+1}] Kategori: {d['category']} | Produk: {d['product']} | Sentimen: {d['sentiment']}\nUlasan: {d['review']}"
        for i, d in enumerate(docs)
    )
    prompt = f"""Anda adalah asisten analisis kepuasan pelanggan. Jawab pertanyaan berikut HANYA
berdasarkan potongan ulasan pelanggan di bawah ini. Jika informasi tidak cukup, katakan demikian.
Sertakan kesimpulan singkat dan sebutkan dokumen mana yang mendukung jawaban Anda.

PERTANYAAN: {question}

KONTEKS ULASAN PELANGGAN:
{context}

JAWABAN:"""
    response = model.generate_content(prompt)
    return response.text

# ----------------------------------------------------------------------
# 4. UI
# ----------------------------------------------------------------------
st.title("🛍️ RAG - Analisis Kepuasan Pelanggan (PRDECT-ID)")
st.caption("Prototipe Retrieval-Augmented Generation untuk UAS Trending Topics on Statistics")

with st.sidebar:
    st.header("⚙️ Pengaturan")
    api_key = st.text_input("Google Gemini API Key", type="password",
                             help="Dapatkan gratis di https://aistudio.google.com/apikey")
    top_k = st.slider("Jumlah dokumen yang di-retrieve", 3, 10, 5)
    st.markdown("---")
    st.write(f"📊 Total dokumen di knowledge base: **{len(df)}**")
    st.write(f"🏷️ Jumlah kategori produk: **{df['Category'].nunique()}**")

question = st.text_input("Tanyakan sesuatu tentang ulasan pelanggan:",
                          placeholder="Contoh: Apakah kualitas produk kategori Kitchen memuaskan pelanggan?")

col1, col2 = st.columns([1, 4])
with col1:
    run = st.button("🔍 Cari Jawaban", type="primary")

if run and question:
    with st.spinner("Mencari dokumen relevan..."):
        docs = retrieve(question, top_k=top_k)

    st.subheader("💬 Jawaban")
    if api_key:
        with st.spinner("Menghasilkan jawaban dengan LLM..."):
            try:
                answer = generate_answer(api_key, question, docs)
                st.write(answer)
            except Exception as e:
                st.error(f"Gagal memanggil LLM: {e}")
    else:
        st.info("Masukkan API key Gemini di sidebar untuk mengaktifkan generation LLM. "
                "Berikut ditampilkan hasil retrieval saja sebagai bukti konsep.")

    st.subheader("📚 Dokumen Referensi (hasil retrieval)")
    for d in docs:
        with st.expander(f"[{d['score']:.3f}] {d['category']} — {d['product'][:60]} ({d['sentiment']})"):
            st.write(d["review"])

st.markdown("---")
st.caption("Dataset: PRDECT-ID (Product Review Dataset for Emotions Classification Tasks in Indonesian)")
