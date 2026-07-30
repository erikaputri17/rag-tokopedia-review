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
st.set_page_config(
    page_title="RAG - Kepuasan Pelanggan PRDECT-ID",
    page_icon="🛍️",
    layout="wide")

MISTRAL_API_KEY = st.secrets["MISTRAL_API_KEY"]

# ----------------------------------------------------------------------
# 1. LOAD & PREPARE DATA (cache agar tidak diproses ulang setiap interaksi)
# ----------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("PRDECT-ID Dataset.csv")

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
def generate_answer(question, docs):
    from mistralai import Mistral
    client = Mistral(api_key=MISTRAL_API_KEY)
    context = "\n\n".join(
        f"[Dok {i+1}] "
        f"Kategori: {d['category']} | "
        f"Produk: {d['product']} | "
        f"Sentimen: {d['sentiment']}\n"
        f"Ulasan: {d['review']}"
        for i, d in enumerate(docs))
    
    prompt = f"""
Anda adalah AI Assistant untuk analisis keputusan pelanggan.

Jawablah pertanyaan pengguna HANYA berdasarkan review yang diberikan.

Jangan menambahkan informasi di luar konteks.

Jika informasi tidak cukup, katakan bahwa data belum mencukupi.

Berikan jawaban dalam bahasa Indonesia.

Sebutkan dokumen mana yang mendukung jawaban.

Jangan menyalin review mentah.

Buatlah jawaban dalam bentuk ringkasan yang mudah dipahami.

Jika ada beberapa review, rangkum menjadi satu kesimpulan.

Jika informasi tidak cukup, katakan bahwa data belum cukup.

PERTANYAAN

{question}

KONTEKS

{context}

JAWABAN
"""
    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content

# ----------------------------------------------------------------------
# 4. UI
# ----------------------------------------------------------------------
st.title("🛍️ RAG - Analisis Kepuasan Pelanggan (PRDECT-ID)")
st.caption("Prototipe Retrieval-Augmented Generation untuk UAS Trending Topics on Statistics")

with st.sidebar:
    st.header("⚙️ Pengaturan")
    top_k = st.slider("Jumlah dokumen yang di-retrieve", 3, 10, 5)
    st.markdown("---")

question = st.text_area("Masukkan pertanyaan",
                          placeholder="Contoh: Apakah pelanggan puas dengan kualitas produk?",
                        height=120)

col1, col2 = st.columns([1, 4])
with col1:
    run = st.button("🔍 Cari Jawaban", type="primary")

if run and question:

    # Retrieval
    with st.spinner("Mencari dokumen..."):
        docs = retrieve(question, top_k)

    # Generation
    with st.spinner("Menyusun jawaban..."):
        answer = generate_answer(question, docs)

    st.subheader("💬 Jawaban")
    st.write(answer)

    st.subheader("📚 Referensi")

    for d in docs:
        with st.expander(
            f"{d['category']} - {d['product']}"
        ):
            st.write(d["review"])

st.markdown("---")
st.caption("Dataset: PRDECT-ID (Product Review Dataset for Emotions Classification Tasks in Indonesian)")
