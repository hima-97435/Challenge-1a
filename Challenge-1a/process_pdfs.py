# process_pdfs.py
from pathlib import Path
import json
import joblib
from vectorizer import extract_features_from_pdf, vectorize

# Load model
model = joblib.load("model.pkl")

def process_pdf_to_json(pdf_path, model):
    entries = extract_features_from_pdf(pdf_path)
    X = [vectorize(e) for e in entries]
    y_pred = model.predict(X)

    result = {"title": "", "outline": []}
    title_found = False

    for entry, label in zip(entries, y_pred):
        if label == "Title" and not title_found:
            result["title"] = entry["text"]
            title_found = True
        elif label in ["H1", "H2", "H3"]:
            result["outline"].append({
                "level": label,
                "text": entry["text"],
                "page": entry["page_num"]
            })

    return result

def process_all_pdfs():
    input_dir = Path("/app/input")
    output_dir = Path("/app/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    for pdf_file in input_dir.glob("*.pdf"):
        result = process_pdf_to_json(pdf_file, model)
        output_file = output_dir / f"{pdf_file.stem}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4)

if __name__ == "__main__":
    process_all_pdfs()
