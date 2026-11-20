from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", BASE_DIR / "chroma_db"))


CHUNK_SIZE = int(os.getenv("CHUNK_SIZE","1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP","200"))

SUPPORTED_PROVIDERS = ("gemini", "openai")
LLM_PROVIDER = os.getenv("LLM_PROVIDER","gemini").strip().lower()

GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")


GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-flash-latest")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-5-mini")

GEMINI_FALLBACK_MODELS = os.getenv("GEMINI_FALLBACK_MODELS", "gemini-flash-lite-latest")
OPENAI_FALLBACK_MODELS = os.getenv("OPENAI_FALLBACK_MODELS", "")

LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "1"))

TOP_K = int(os.getenv("TOP_K", "3"))
CORS_ORIGINS = [
    o.strip()
    for o in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
    if o.strip()
]

COLLECTION_NAME = os.getenv("COLLECTION_NAME", f"documents_{LLM_PROVIDER}")


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(
            f"{name} is not set. Copy .env.example to .env and add your key "
            f"(LLM_PROVIDER is currently '{LLM_PROVIDER}')."
        )
    return value


def get_embeddings() -> Embeddings:
    if LLM_PROVIDER == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=GEMINI_EMBED_MODEL,
            google_api_key=_require_env("GOOGLE_API_KEY"),
        )

    if LLM_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=OPENAI_EMBED_MODEL,
            api_key=_require_env("OPENAI_API_KEY"),
        )

    raise ValueError(
        f"Unsupported LLM_PROVIDER '{LLM_PROVIDER}'. "
        f"Choose one of: {', '.join(SUPPORTED_PROVIDERS)}."
    )


def _split_models(value: str) -> list[str]:
    return [m.strip() for m in value.split(",") if m.strip()]


def _build_chat_model(model: str) -> BaseChatModel:
    if LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=_require_env("GOOGLE_API_KEY"),
            max_retries=LLM_MAX_RETRIES,
        )

    if LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            api_key=_require_env("OPENAI_API_KEY"),
            max_retries=LLM_MAX_RETRIES,
        )

    raise ValueError(
        f"Unsupported LLM_PROVIDER '{LLM_PROVIDER}'. "
        f"Choose one of: {', '.join(SUPPORTED_PROVIDERS)}."
    )


def get_llm() -> Runnable:
    if LLM_PROVIDER == "gemini":
        primary, fallbacks = GEMINI_CHAT_MODEL, _split_models(GEMINI_FALLBACK_MODELS)
    elif LLM_PROVIDER == "openai":
        primary, fallbacks = OPENAI_CHAT_MODEL, _split_models(OPENAI_FALLBACK_MODELS)
    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER '{LLM_PROVIDER}'. "
            f"Choose one of: {', '.join(SUPPORTED_PROVIDERS)}."
        )
    fallbacks = [m for m in fallbacks if m != primary]
    llm = _build_chat_model(primary)
    if not fallbacks:
        return llm
    return llm.with_fallbacks([_build_chat_model(m) for m in fallbacks])


def get_vectorstore(embeddings: Embeddings | None = None) -> Chroma:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings or get_embeddings(),
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )