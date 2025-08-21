
import os, json, argparse, csv
from pathlib import Path
from typing import List, Dict, Any

from plant_rag_system import PlantRAGSystem

def load_questions(path: Path) -> List[Dict[str, Any]]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items

def heuristic_scores(gt: str, pred: str) -> Dict[str, float]:
    # F1 lexical simple (casefold + mots > 2 chars)
    import re
    tok = lambda s: [w for w in re.findall(r"\w+", s.casefold()) if len(w) > 2]
    g, p = set(tok(gt)), set(tok(pred))
    if not g or not p:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    inter = g & p
    prec = len(inter) / len(p)
    rec = len(inter) / len(g)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}

def try_ragas_evaluation(rows: List[Dict[str, Any]]):
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        # ragas llm/embeddings adapters (API can vary by version)
        try:
            from ragas.llms import OpenAI as RagasOpenAI
            from ragas.embeddings import LangchainEmbeddings
        except Exception:
            # Some versions expose the same names under ragas.*
            from ragas.llms import OpenAI as RagasOpenAI
            from ragas.embeddings import LangchainEmbeddings

        from langchain_community.embeddings import HuggingFaceEmbeddings

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("⚠️  OPENAI_API_KEY non défini : RAGAS sautée (utilisez la grille heuristique).")
            return None

        llm = RagasOpenAI(model="gpt-4o-mini")  # judge
        emb = LangchainEmbeddings(HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base",
                                                        encode_kwargs={"normalize_embeddings": True}))

        dataset = {
            "question": [r["question"] for r in rows],
            "answer": [r["answer"] for r in rows],
            "contexts": [r["contexts"] for r in rows],
            "ground_truth": [r.get("ground_truth","") for r in rows],
        }
        ds = Dataset.from_dict(dataset)
        result = evaluate(ds, metrics=[faithfulness, answer_relevancy, context_precision, context_recall], llm=llm, embeddings=emb)
        return result
    except Exception as e:
        print(f"⚠️  RAGAS indisponible ou erreur: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Évalue un RAG botanique avec RAGAS (si dispo) + heuristiques.")
    parser.add_argument("--dataset_path", type=str, default="dataset", help="Dossier contenant les .md")
    parser.add_argument("--questions_file", type=str, default="eval_questions.jsonl", help="Fichier JSONL de questions")
    parser.add_argument("--persist_dir", type=str, default="./chroma_db_plants_eval", help="Répertoire de persistance Chroma")
    args = parser.parse_args()

    questions_path = Path(args.questions_file)
    if not questions_path.exists():
        raise SystemExit(f"Fichier de questions introuvable: {questions_path}")

    print("🌱 Chargement du RAG...")
    rag = PlantRAGSystem(dataset_path=args.dataset_path, persist_dir=args.persist_dir, collection_name="plants_eval_v1")
    rag.load_plant_files()
    rag.create_vectorstore()

    print("🧪 Réponses aux questions...")
    items = load_questions(questions_path)
    rows = []
    for q in items:
        out = rag.answer_question(q["question"], k=q.get("k",8), verbose=True)
        row = {
            "id": q["id"],
            "question": q["question"],
            "ground_truth": q.get("ground_truth", ""),
            "expected_doc_ids": q.get("doc_ids", []),
            "answer": out["answer"],
            "contexts": out["contexts"],
            "metadatas": out["metadatas"],
        }
        rows.append(row)

    # Créer le dossier test_results s'il n'existe pas
    test_results_dir = Path("test_results")
    test_results_dir.mkdir(exist_ok=True)
    
    # Sauvegarde brute
    out_json = test_results_dir / "eval_results_raw.json"
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"💾 Résultats bruts: {out_json.resolve()}")

    # Heuristiques minimales
    print("📏 Évaluation heuristique...")
    heuristics = []
    for r in rows:
        h = heuristic_scores(r.get("ground_truth",""), r["answer"])
        # context doc hit-rate: part des contexts venant d'un doc attendu (si doc_ids fournis)
        expected = set(r.get("expected_doc_ids") or [])
        if expected:
            hits = sum(1 for m in r["metadatas"] if (m.get("doc_id") in expected))
            hit_rate = hits / max(1, len(r["metadatas"]))
        else:
            hit_rate = 0.0
        heuristics.append({
            "id": r["id"],
            "question": r["question"],
            "lex_f1": round(h["f1"], 3),
            "lex_precision": round(h["precision"], 3),
            "lex_recall": round(h["recall"], 3),
            "context_hit_rate": round(hit_rate, 3),
        })

    out_csv = test_results_dir / "heuristic_report.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(heuristics[0].keys()))
        w.writeheader()
        w.writerows(heuristics)
    print(f"💾 Rapport heuristique: {out_csv.resolve()}")

    # RAGAS (si dispo + clé OpenAI)
    print("🧮 Évaluation RAGAS...")
    ragas_result = try_ragas_evaluation(rows)
    if ragas_result is not None:
        try:
            # ragas_result peut offrir .to_pandas() selon la version
            df = ragas_result.to_pandas() if hasattr(ragas_result, "to_pandas") else None
            if df is not None:
                df.to_csv(test_results_dir / "ragas_report.csv", index=False, encoding="utf-8")
                print("💾 Rapport RAGAS: test_results/ragas_report.csv")
            else:
                # sauvegarde JSON brute
                (test_results_dir / "ragas_report.json").write_text(json.dumps(ragas_result, ensure_ascii=False, indent=2), encoding="utf-8")
                print("💾 Rapport RAGAS: test_results/ragas_report.json")
        except Exception as e:
            print(f"⚠️  Sauvegarde RAGAS: {e}")

    print("✅ Terminé.")

if __name__ == "__main__":
    main()
