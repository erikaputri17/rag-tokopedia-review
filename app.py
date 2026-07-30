"""
RAG Streamlit App
Analisis Kepuasan Pelanggan PRDECT-ID
"""

import streamlit as st
import pandas as pd
import re
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import google.generativeai as genai

# ======================================================
# KONFIGURASI HALAMAN
# ======================================================

st.set_page_config(
    page_title="RAG - Analisis Kepuasan Pelanggan",
    page_icon="🛍️",
    layout="wide"
)

# ======================================================
# API KEY GEMINI
# ======================================================

GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel("gemini-1.5-flash")

# ======================================================
# LOAD DATASET
# ======================================================

@st.cache_data
def load_data():

    df = pd.read_csv("PRDECT-ID_Dataset.csv")

    def clean_text(text):

        text = str(text).lower()

        text = re.sub(r"http\S+|www\S+", " ", text)

        text = re.sub(r"\d+", " ", text)

        text = re.sub(r"[^\w\s]", " ", text)

        text = text.translate(
            str.maketrans("", "", string.punctuation)
        )

        text = re.sub(r"\s+", " ", text).strip()

        return text

    df["clean_text"] = df["Customer Review"].apply(clean_text)

    df["kb_text"] = (
        "Kategori : "
        + df["Category"].astype(str)
        + "\nProduk : "
        + df["Product Name"].astype(str)
        + "\nSentimen : "
        + df["Sentiment"].astype(str)
        + "\nReview : "
        + df["Customer Review"].astype(str)
    )

    return df


df = load_data()

# ======================================================
# MEMBANGUN TF-IDF
# ======================================================

@st.cache_resource
def build_index(df):

    vectorizer = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1,2)
    )

    matrix = vectorizer.fit_transform(
        df["clean_text"]
    )

    return vectorizer, matrix


vectorizer, tfidf_matrix = build_index(df)

# ======================================================
# RETRIEVAL
# ======================================================

def retrieve(query, top_k=5):

    query = query.lower()

    query = re.sub(r"[^\x00-\x7F]+", " ", query)

    query_vector = vectorizer.transform([query])

    similarity = cosine_similarity(
        query_vector,
        tfidf_matrix
    ).ravel()

    index = similarity.argsort()[::-1][:top_k]

    documents = []

    for i in index:

        documents.append({

            "doc_id": int(i),

            "score": float(similarity[i]),

            "category": df.iloc[i]["Category"],

            "product": df.iloc[i]["Product Name"],

            "review": df.iloc[i]["Customer Review"],

            "sentiment": df.iloc[i]["Sentiment"]

        })

    return documents


# ======================================================
# GENERATION (GEMINI)
# ======================================================

def generate_answer(question, retrieved_docs):

    context = "\n\n".join(

        [

            f"[Dok {i+1}] "

            f"Kategori: {d['category']} | "

            f"Produk: {d['product']} | "

            f"Sentimen: {d['sentiment']}\n"

            f"Ulasan: {d['review']}"

            for i, d in enumerate(retrieved_docs)

        ]

    )

    prompt = f"""
Anda adalah asisten analisis kepuasan pelanggan.

Jawablah pertanyaan HANYA berdasarkan review pelanggan di bawah ini.

Jangan menggunakan pengetahuan di luar konteks.

Jika informasi tidak cukup, katakan bahwa data tidak mencukupi.

Buat jawaban dalam bahasa Indonesia.

Ringkas hasil review menjadi beberapa paragraf.

Jangan menyalin review satu per satu.

Di bagian akhir tuliskan dokumen mana yang menjadi dasar jawaban.

PERTANYAAN

{question}

KONTEKS REVIEW

{context}

JAWABAN
"""

    response = model.generate_content(prompt)

    return response.text


# ======================================================
# RAG PIPELINE
# ======================================================

def rag_pipeline(question, top_k):

    docs = retrieve(question, top_k)

    answer = generate_answer(question, docs)

    return answer, docs

# ======================================================
# USER INTERFACE
# ======================================================

st.title("🛍️ RAG - Analisis Kepuasan Pelanggan PRDECT-ID")

st.caption(
    "Retrieval-Augmented Generation (RAG) menggunakan TF-IDF, "
    "Cosine Similarity, dan Google Gemini"
)

# ===========================
# Sidebar
# ===========================

with st.sidebar:

    st.header("⚙️ Pengaturan")

    top_k = st.slider(
        "Jumlah dokumen yang diambil",
        min_value=3,
        max_value=10,
        value=5
    )

    st.markdown("---")

    st.metric(
        "Jumlah Review",
        len(df)
    )

    st.metric(
        "Kategori Produk",
        df["Category"].nunique()
    )

    st.metric(
        "Review Positif",
        (df["Sentiment"] == "Positive").sum()
    )

    st.metric(
        "Review Negatif",
        (df["Sentiment"] == "Negative").sum()
    )

# ===========================
# Input Pertanyaan
# ===========================

question = st.text_area(
    "Masukkan pertanyaan",
    placeholder="Contoh: Bagaimana kepuasan pelanggan terhadap kategori Kitchen?",
    height=120
)

if st.button("🔍 Cari Jawaban", use_container_width=True):

    if question.strip() == "":
        st.warning("Silakan masukkan pertanyaan terlebih dahulu.")

    else:

        with st.spinner("Mencari dokumen yang relevan..."):

            docs = retrieve(question, top_k)

        with st.spinner("Menyusun jawaban menggunakan Gemini..."):

            try:

                answer = generate_answer(question, docs)

                st.subheader("💬 Jawaban AI")

                st.success(answer)

            except Exception as e:

                st.error(f"Gagal menghasilkan jawaban.\n\n{e}")

        st.divider()

        st.subheader("📚 Dokumen Referensi")

        for i, d in enumerate(docs):

            with st.expander(
                f"Dokumen {i+1} | Similarity = {d['score']:.3f}"
            ):

                st.write(
                    f"**Kategori :** {d['category']}"
                )

                st.write(
                    f"**Produk :** {d['product']}"
                )

                st.write(
                    f"**Sentimen :** {d['sentiment']}"
                )

                st.write("**Isi Review**")

                st.write(d["review"])

st.markdown("---")

st.caption(
    "Dataset : PRDECT-ID (Product Review Dataset for Emotion Classification Tasks in Indonesian)"
)
