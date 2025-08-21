import os, re, glob, hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path
import yaml
from dotenv import load_dotenv

from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Splitters
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# ---- Utils ----

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def extract_front_matter(md: str) -> (Dict[str, Any], str):
    """
    Extrait proprement le front-matter YAML et renvoie (metadata, markdown_sans_front_matter)
    """
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", md, flags=re.DOTALL)
    if not m:
        return {}, md
    meta_block, body = m.group(1), m.group(2)
    meta = yaml.safe_load(meta_block) or {}
    return meta, body

def token_len(s: str) -> int:
    # Simple approx pour éviter tiktoken si tu veux rester ultra léger
    # (ou remplace par tiktoken pour la vraie prod)
    return max(1, len(s.split()))

# ---- RAG ----

class PlantRAGSystem:
    def __init__(
        self,
        dataset_path: str = "dataset",
        model_name: str = "intfloat/multilingual-e5-base",
        persist_dir: str = "./chroma_db_plants",
        collection_name: str = "plants_v1",
    ):
        load_dotenv()
        self.dataset_path = dataset_path
        self.persist_dir = persist_dir
        self.collection_name = collection_name

        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},                 # ajuste à "cuda" si GPU
            encode_kwargs={"normalize_embeddings": True},   # très important
        )

        self.vectorstore: Optional[Chroma] = None
        self.documents: List[Document] = []

    # ---------- Ingestion ----------

    def load_plant_files(self) -> List[Document]:
        md_files = sorted(glob.glob(os.path.join(self.dataset_path, "*.md")))
        docs: List[Document] = []

        for fp in md_files:
            try:
                text = Path(fp).read_text(encoding="utf-8")
                meta, body = extract_front_matter(text)

                # Conserve les titres Markdown (on NE convertit PAS en HTML)
                content = body.strip()

                # Métadonnées minimales + checksum (traçabilité)
                mdict = {
                    "source": fp,
                    "doc_id": meta.get("id", Path(fp).stem),
                    "title": meta.get("title", ""),
                    "scientific_name": meta.get("scientific_name", ""),
                    "common_names": meta.get("common_names", []),
                    "family": meta.get("family", ""),
                    "difficulty": meta.get("difficulty", None),
                    "light": meta.get("light", ""),
                    "water": meta.get("water", ""),
                    "tags": meta.get("tags", []),
                    "lang": meta.get("lang", "fr"),
                    "license": meta.get("license", ""),
                    "version": meta.get("version", ""),
                    "checksum": sha256_text(text),
                }

                docs.append(Document(page_content=content, metadata=mdict))
                print(f"✓ Chargé: {mdict['title'] or fp}")
            except Exception as e:
                print(f"✗ Erreur {fp}: {e}")

        self.documents = docs
        return docs

    def _clean_metadata_for_chroma(self, metadata: dict) -> dict:
        """
        Nettoie les métadonnées pour ChromaDB en convertissant les listes en chaînes
        """
        cleaned = {}
        for key, value in metadata.items():
            if isinstance(value, list):
                # Convertir les listes en chaînes séparées par des virgules
                cleaned[key] = ", ".join(str(item) for item in value)
            elif isinstance(value, (str, int, float, bool)) or value is None:
                # Garder les types compatibles avec ChromaDB
                cleaned[key] = value
            else:
                # Convertir les autres types en chaînes
                cleaned[key] = str(value)
        return cleaned

    def _split_markdown(self, docs: List[Document]) -> List[Document]:
        """
        1) Split par titres Markdown (préserve la sémantique)
        2) Puis split "caractère/tokens" avec overlap
        """
        out: List[Document] = []
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
            ]
        )

        for d in docs:
            # 1) éclater par titres
            header_docs = header_splitter.split_text(d.page_content)
            # réinjecter les métadonnées d'origine et les nettoyer
            header_docs = [
                Document(
                    page_content=hd.page_content, 
                    metadata=self._clean_metadata_for_chroma({**d.metadata, **hd.metadata})
                )
                for hd in header_docs
            ]

            # 2) split sémantique court (200–400 tokens)
            char_splitter = RecursiveCharacterTextSplitter(
                chunk_size=350,            # ~200–400 tokens en FR
                chunk_overlap=80,
                separators=["\n\n", "\n", " ", ""],
                length_function=token_len,
            )
            out.extend(char_splitter.split_documents(header_docs))

        return out

    def create_vectorstore(self) -> Chroma:
        if not self.documents:
            self.load_plant_files()

        chunks = self._split_markdown(self.documents)
        print(f"✓ {len(chunks)} chunks générés")

        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_dir,
            collection_name=self.collection_name,
        )
        # Persistence automatique avec Chroma 0.4.x - plus besoin d'appel explicite
        print("✓ Base vectorielle persistée")
        return self.vectorstore

    # ---------- Recherche & QA ----------

    def retriever(self, k: int = 4, score_threshold: float = 0.2):
        if not self.vectorstore:
            raise ValueError("Vectorstore non initialisée. Appelle create_vectorstore().")

        return self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                "fetch_k": max(10, 3 * k),
                "lambda_mult": 0.5,
                "score_threshold": score_threshold,  # filtrage du bruit
            },
        )

    def search_plants(self, query: str, k: int = 5) -> List[Document]:
        return self.retriever(k=k).invoke(query)

    def answer_question(self, question: str, k: int = 4, model: Optional[str] = None) -> str:
        if not self.vectorstore:
            raise ValueError("Vectorstore non initialisée.")
        docs = self.search_plants(question, k=k)

        context_blocks = []
        for i, d in enumerate(docs, 1):
            src = d.metadata.get("source", "")
            title = d.metadata.get("title", "")
            header = d.metadata.get("h2") or d.metadata.get("h3") or ""
            context_blocks.append(
                f"[{i}] {title} — {header}\nSOURCE: {src}\n\n{d.page_content}"
            )
        context = "\n\n---\n\n".join(context_blocks)

        prompt = f"""
Tu es un assistant botanique. Réponds en français, de façon précise et concise.
Utilise UNIQUEMENT les informations du contexte. SI tu n'es pas sûr, dis-le.

Question:
{question}

Contexte (extraits avec sources numérotées):
{context}

Consignes:
- Donne la réponse synthétique d'abord (3–6 lignes).
- Ajoute ensuite une section "Sources" listant les numéros et titres : [1], [2], ...
- Si des précautions ou contre-exemples existent, cite-les.

Réponse:
""".strip()

        # LLM si dispo, sinon réponse extractive simple
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                llm = ChatOpenAI(model=model or "gpt-4o-mini", temperature=0)
                rsp = llm.invoke(prompt)
                return rsp.content
            else:
                # Fallback: renvoie un extrait + sources
                head = "\n\n".join(b[:500] for b in context_blocks[:2])
                srcs = "\n".join(
                    f"- [{i+1}] {d.metadata.get('title','')} ({d.metadata.get('source','')})"
                    for i, d in enumerate(docs)
                )
                return f"{head}\n\nSources:\n{srcs}"
        except Exception as e:
            return f"Erreur génération: {e}"

def main():
    print("🌱 Init RAG Plantes (amélioré)")
    rag = PlantRAGSystem()
    print("📚 Chargement .md ...")
    docs = rag.load_plant_files()
    print(f"✓ {len(docs)} fiches")

    print("🔍 Construction vecteur ...")
    rag.create_vectorstore()

    print("🧪 Recherche: 'plante facile pour débutant intérieur'")
    for d in rag.search_plants("plante facile pour débutant intérieur", k=3):
        print("-", d.metadata.get("title"), "| diff:", d.metadata.get("difficulty"))

    print("❓ QA")
    q = "Comment entretenir une plante araignée ?"
    print("Q:", q)
    print(rag.answer_question(q))

if __name__ == "__main__":
    main()
