
import os
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


@st.cache_resource
def load_embedding_model():
    model_name = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    return SentenceTransformer(model_name)


def nonempty(value):
    value = str(value).strip().lower()
    return value not in {
        "", "not listed", "nan", "none",
        "n/a", "not available"
    }


def build_documents(products, additives):
    documents = []

    for index, row in products.iterrows():
        parts = []

        for col in products.columns:
            value = row.get(col, "")
            if nonempty(value):
                parts.append(f"{col}: {value}")

        documents.append({
            "kind": "product",
            "index": int(index),
            "title": str(row.get("Item name", "Product")),
            "text": "\n".join(parts)
        })

    for index, row in additives.iterrows():
        parts = []

        for col in additives.columns:
            value = row.get(col, "")
            if nonempty(value):
                parts.append(f"{col}: {value}")

        documents.append({
            "kind": "additive",
            "index": int(index),
            "title": str(
                row.get("Ingredient_or_Additive", "Additive")
            ),
            "text": "\n".join(parts)
        })

    return documents


@st.cache_resource
def encode_documents(document_texts):
    model = load_embedding_model()
    vectors = model.encode(
        list(document_texts),
        normalize_embeddings=True,
        show_progress_bar=False
    )
    return model, vectors


def build_retriever(products, additives):
    documents = build_documents(products, additives)
    texts = tuple(doc["text"] for doc in documents)

    model, vectors = encode_documents(texts)

    return {
        "documents": documents,
        "model": model,
        "vectors": vectors
    }


def retrieve_context(
    query,
    retriever,
    product=None,
    top_k=6
):
    documents = retriever["documents"]
    model = retriever["model"]
    vectors = retriever["vectors"]

    query_vector = model.encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False
    )

    scores = cosine_similarity(query_vector, vectors)[0]
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: float(scores[i]),
        reverse=True
    )

    chosen = []

    # Always include the active product when available.
    if product is not None:
        active_index = int(product.name)

        for doc in documents:
            if (
                doc["kind"] == "product"
                and doc["index"] == active_index
            ):
                chosen.append(doc)
                break

    for index in ranked_indices:
        doc = documents[index]

        if doc in chosen:
            continue

        chosen.append(doc)

        if len(chosen) >= top_k + (
            1 if product is not None else 0
        ):
            break

    return "\n\n---\n\n".join(
        f"[{doc['kind'].upper()} | {doc['title']}]\n"
        f"{doc['text']}"
        for doc in chosen
    )
