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

    def retriever(self, k: int = 8, score_threshold: float = 0.15):
        if not self.vectorstore:
            raise ValueError("Vectorstore non initialisée. Appelle create_vectorstore().")

        return self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                "fetch_k": max(20, 4 * k),  # Augmenté pour plus de couverture
                "lambda_mult": 0.7,  # Plus de diversité dans les résultats
                "score_threshold": score_threshold,  # Seuil plus bas pour plus de résultats
            },
        )

    def search_plants(self, query: str, k: int = 8) -> List[Document]:
        return self.retriever(k=k).invoke(query)

    def _expand_query(self, question: str) -> str:
        """Expansion intelligente de la requête avec synonymes botaniques"""
        expansions = {
            "romarin": "Rosmarinus officinalis romarin",
            "basilic": "Ocimum basilicum basilic",
            "menthe": "Mentha spicata menthe",
            "lavande": "Lavandula angustifolia lavande",
            "monstera": "Monstera deliciosa plante gruyère",
            "ficus": "Ficus lyrata figuier lyre",
            "aloe": "Aloe vera aloe",
            "sansevieria": "Sansevieria trifasciata langue de belle-mère",
            "chlorophytum": "Chlorophytum comosum plante araignée",
            "lierre": "Hedera helix lierre",
            "température": "température chaleur froid",
            "arrosage": "arrosage eau humidité",
            "lumière": "lumière soleil exposition éclairage",
            "multiplication": "multiplication bouture division semis",
            "rempotage": "rempotage pot substrat",
            "floraison": "floraison fleur épanouissement",
            "envahissante": "envahissante expansion contrôle",
            "sécheresse": "sécheresse résistance tolérance"
        }
        
        expanded = question.lower()
        for term, expansion in expansions.items():
            if term in expanded:
                expanded += " " + expansion
        
        return expanded

    def _filter_relevant_docs(self, docs: List[Document], question: str) -> List[Document]:
        """Filtrage intelligent des documents par pertinence et métadonnées"""
        if not docs:
            return docs
            
        # Score de pertinence basé sur les métadonnées
        scored_docs = []
        question_lower = question.lower()
        
        for doc in docs:
            score = 0
            metadata = doc.metadata
            
            # Bonus pour les correspondances exactes dans les métadonnées
            if any(term in question_lower for term in metadata.get("common_names", [])):
                score += 3
            if any(term in question_lower for term in metadata.get("tags", [])):
                score += 2
            if metadata.get("scientific_name", "").lower() in question_lower:
                score += 4
            if metadata.get("title", "").lower() in question_lower:
                score += 3
                
            # Bonus pour les familles botaniques pertinentes
            if "aromatique" in question_lower and metadata.get("family") in ["Lamiaceae", "Apiaceae"]:
                score += 2
            if "intérieur" in question_lower and "plante d'intérieur" in metadata.get("tags", []):
                score += 2
                
            scored_docs.append((doc, score))
        
        # Trier par score et retourner les meilleurs
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored_docs[:len(docs)]]

    def answer_question(self, question: str, k: int = 8, model: Optional[str] = None, verbose: bool = False) -> Dict[str, Any]:
        if not self.vectorstore:
            raise ValueError("Vectorstore non initialisée.")
        
        # Expansion de la requête pour améliorer la recherche
        expanded_query = self._expand_query(question)
        if verbose:
            print(f"🔍 Requête originale: {question}")
            print(f"🔍 Requête étendue: {expanded_query}")
        
        # Recherche avec plus de résultats
        docs = self.search_plants(expanded_query, k=k*2)
        if verbose:
            print(f"📚 {len(docs)} documents trouvés initialement")
        
        # Filtrage intelligent des documents les plus pertinents
        docs = self._filter_relevant_docs(docs, question)
        if verbose:
            print(f"🎯 {len(docs)} documents filtrés et classés")
            for i, doc in enumerate(docs[:3]):
                print(f"  [{i+1}] {doc.metadata.get('title', 'Sans titre')} (score: {doc.metadata.get('doc_id', 'N/A')})")

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
Tu es un assistant botanique expert. Réponds EXCLUSIVEMENT en français, de façon PRÉCISE et CONCISE.
Utilise UNIQUEMENT les informations du contexte fourni. Si une information n'est pas dans le contexte, NE l'invente PAS.

Question: {question}

Contexte (extraits avec sources numérotées):
{context}

INSTRUCTIONS STRICTES:
1. Réponse principale: 2-4 lignes maximum, directe et factuelle
2. Sources: Liste numérotée [1], [2], etc. avec les titres des documents utilisés
3. Précautions: Seulement si mentionnées dans le contexte
4. Format: Réponse + Sources + Précautions (si applicable)

RÈGLES:
- Pas d'informations générales non présentes dans le contexte
- Pas de répétitions
- Pas de phrases vides ou redondantes
- Utilise les termes exacts du contexte

Réponse:
""".strip()

        # LLM si dispo, sinon réponse extractive simple
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                llm = ChatOpenAI(model=model or "gpt-4o-mini", temperature=0)
                rsp = llm.invoke(prompt)
                answer = rsp.content
            else:
                # Fallback: renvoie un extrait + sources
                head = "\n\n".join(b[:500] for b in context_blocks[:2])
                srcs = "\n".join(
                    f"- [{i+1}] {d.metadata.get('title','')} ({d.metadata.get('source','')})"
                    for i, d in enumerate(docs)
                )
                answer = f"{head}\n\nSources:\n{srcs}"
        except Exception as e:
            answer = f"Erreur génération: {e}"

        # Post-processing pour améliorer la qualité de la réponse
        cleaned_answer = self._clean_response(answer, docs)
        
        # Retourne le format attendu par l'évaluation
        return {
            "answer": cleaned_answer,
            "contexts": [d.page_content for d in docs],
            "metadatas": [d.metadata for d in docs]
        }

    def _clean_response(self, answer: str, docs: List[Document]) -> str:
        """Nettoyage et amélioration de la réponse générée"""
        if not answer:
            return "Aucune réponse générée."
            
        # Supprimer les répétitions et phrases vides
        lines = answer.split('\n')
        cleaned_lines = []
        seen_content = set()
        
        for line in lines:
            line = line.strip()
            if not line or line in ['Réponse:', 'Sources:', 'Précautions:']:
                continue
                
            # Éviter les répétitions
            content_key = line.lower().replace(' ', '')[:50]
            if content_key not in seen_content:
                cleaned_lines.append(line)
                seen_content.add(content_key)
        
        # Reconstruire la réponse
        if not cleaned_lines:
            return "Aucune information pertinente trouvée dans le contexte."
            
        return '\n'.join(cleaned_lines)

    def diagnose_search(self, question: str, k: int = 8) -> Dict[str, Any]:
        """Diagnostic détaillé de la recherche pour une question donnée"""
        if not self.vectorstore:
            raise ValueError("Vectorstore non initialisée.")
            
        # Test de la recherche brute
        raw_docs = self.search_plants(question, k=k*2)
        
        # Test avec expansion de requête
        expanded_query = self._expand_query(question)
        expanded_docs = self.search_plants(expanded_query, k=k*2)
        
        # Test avec filtrage intelligent
        filtered_docs = self._filter_relevant_docs(expanded_docs, question)
        
        return {
            "question_originale": question,
            "question_etendue": expanded_query,
            "recherche_brute": {
                "count": len(raw_docs),
                "titles": [d.metadata.get("title", "Sans titre") for d in raw_docs[:5]]
            },
            "recherche_etendue": {
                "count": len(expanded_docs),
                "titles": [d.metadata.get("title", "Sans titre") for d in expanded_docs[:5]]
            },
            "recherche_filtree": {
                "count": len(filtered_docs),
                "titles": [d.metadata.get("title", "Sans titre") for d in filtered_docs[:5]],
                "scores": [d.metadata.get("doc_id", "N/A") for d in filtered_docs[:5]]
            }
        }

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
