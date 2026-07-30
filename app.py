import streamlit as st
import pandas as pd
import numpy as np
import re
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from mistralai import Mistral

st.set_page_config(
    page_title="RAG Analisis Kepuasan Pelanggan",
    page_icon="🛍️",
    layout="wide"
)

MISTRAL_API_KEY = st.secrets["MISTRAL_API_KEY"]

client = Mistral(
    api_key=MISTRAL_API_KEY
)

@st.cache_data
def load_data():

    df = pd.read_csv("PRDECT-ID_Dataset.csv")

    return df

df = load_data()

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"http\S+", " ", text)

    text = re.sub(r"\d+", " ", text)

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    text = re.sub(r"\s+", " ", text)

    return text.strip()

df["clean_text"] = df["Customer Review"].apply(clean_text)

df["kb_text"] = (

    "Kategori : "

    + df["Category"]

    + "\nProduk : "

    + df["Product Name"]

    + "\nSentimen : "

    + df["Sentiment"]

    + "\nReview : "

    + df["Customer Review"]

)

@st.cache_resource
def build_index():

    vectorizer = TfidfVectorizer(

        max_features=8000,

        ngram_range=(1,2)

    )

    matrix = vectorizer.fit_transform(

        df["clean_text"]

    )

    return vectorizer, matrix

vectorizer, tfidf_matrix = build_index()

# ==========================================================
# RETRIEVAL
# ==========================================================

def retrieve(query, top_k=5):

    # Bersihkan query
    query = clean_text(query)

    # Ubah menjadi vector TF-IDF
    query_vector = vectorizer.transform([query])

    # Hitung cosine similarity
    similarity = cosine_similarity(
        query_vector,
        tfidf_matrix
    ).flatten()

    # Ambil index dengan similarity terbesar
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

# ==========================================================
# GENERATE ANSWER
# ==========================================================

def generate_answer(question, docs):

    context = ""

    for i, d in enumerate(docs):

        context += f"""
Dokumen {i+1}

Kategori : {d['category']}

Produk : {d['product']}

Sentimen : {d['sentiment']}

Review :
{d['review']}

--------------------------------------

"""

    prompt = f"""
Anda adalah AI Assistant yang bertugas menganalisis kepuasan pelanggan e-commerce Indonesia.

Jawablah pertanyaan pengguna HANYA berdasarkan review yang diberikan.

Jangan menggunakan pengetahuan di luar review.

Jika informasi belum cukup, katakan bahwa data review belum mencukupi.

Buat jawaban dalam bahasa Indonesia yang natural.

Jangan menyalin review satu per satu.

Rangkum review menjadi sebuah analisis.

Di bagian akhir tuliskan dokumen mana yang menjadi referensi jawaban.

==============================

PERTANYAAN

{question}

==============================

REVIEW

{context}

==============================

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

# ==========================================================
# RAG PIPELINE
# ==========================================================

def rag(question, top_k):

    docs = retrieve(question, top_k)

    answer = generate_answer(question, docs)

    return answer, docs

st.title("🛍️ RAG - Analisis Kepuasan Pelanggan PRDECT-ID")

st.markdown("""
Prototype **Retrieval-Augmented Generation (RAG)** menggunakan
**TF-IDF + Cosine Similarity + Mistral AI**

Dataset:
PRDECT-ID (Product Review Dataset for Emotion Classification Tasks in Indonesian)
""")

with st.sidebar:

    st.header("⚙️ Pengaturan")

    top_k = st.slider(

        "Jumlah dokumen yang diambil",

        min_value=3,

        max_value=10,

        value=5

    )

    st.markdown("---")

    st.subheader("📊 Informasi Dataset")

    st.write(f"Jumlah Review : **{len(df)}**")

    st.write(f"Kategori Produk : **{df['Category'].nunique()}**")

    st.write(f"Review Positif : **{(df['Sentiment']=='Positive').sum()}**")

    st.write(f"Review Negatif : **{(df['Sentiment']=='Negative').sum()}**")

    st.markdown("---")

    st.subheader("📂 Daftar Kategori")

    kategori = sorted(df["Category"].dropna().unique())

    for k in kategori:

        jumlah = len(df[df["Category"] == k])

        st.write(f"• {k} ({jumlah})")

    st.markdown("---")

    st.subheader("💡 Contoh Pertanyaan")

    st.write("• Bagaimana kepuasan pelanggan pada kategori Beauty?")

    st.write("• Apa keluhan pelanggan terhadap produk Electronics?")

    st.write("• Produk apa yang paling banyak mendapat review positif?")

    st.write("• Mengapa pelanggan memberikan review negatif?")

    st.write("• Apa kelebihan produk kategori Kitchen?")

question = st.text_area(

    "Masukkan pertanyaan",

    height=120,

    placeholder="Contoh: Bagaimana kepuasan pelanggan terhadap kategori Beauty?"

)

run = st.button(

    "🔍 Cari Jawaban",

    use_container_width=True

)

if run:

    if question == "":

        st.warning("Silakan masukkan pertanyaan terlebih dahulu.")

    else:

        with st.spinner("🔍 Mencari dokumen yang relevan..."):

            docs = retrieve(

                question,

                top_k

            )

        with st.spinner("🤖 Menyusun jawaban menggunakan Mistral..."):

            try:

                answer = generate_answer(

                    question,

                    docs

                )

                st.success("Jawaban berhasil dibuat.")

                st.subheader("💬 Jawaban AI")

                st.write(answer)

            except Exception as e:

                st.error(f"Gagal menghasilkan jawaban\n\n{e}")

        st.markdown("---")

        st.subheader("📚 Referensi Dokumen")

        for i, d in enumerate(docs):

            with st.expander(

                f"Dokumen {i+1} | Similarity {d['score']:.3f}"

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

                st.write(

                    d["review"]

                )

st.markdown("---")

st.caption(

    "Universitas Islam Indonesia | Mata Kuliah Trending Topics on Statistics | Prototype Retrieval-Augmented Generation"

)
