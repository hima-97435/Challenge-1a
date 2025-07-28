# vectorizer.py
import fitz  # pymupdf
import re

def extract_features_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    entries = []
    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                line_text = ""
                font_sizes = []
                x0s, y0s, x1s, y1s = [], [], [], []
                for span in line["spans"]:
                    text = span["text"].strip()
                    if not text:
                        continue
                    line_text += " " + text
                    font_sizes.append(span["size"])
                    x0s.append(span["bbox"][0])
                    y0s.append(span["bbox"][1])
                    x1s.append(span["bbox"][2])
                    y1s.append(span["bbox"][3])
                if not line_text.strip():
                    continue
                entry = {
                    "text": line_text.strip(),
                    "font_size": sum(font_sizes) / len(font_sizes),
                    "x0": min(x0s),
                    "y0": min(y0s),
                    "x1": max(x1s),
                    "y1": max(y1s),
                    "width": max(x1s) - min(x0s),
                    "height": max(y1s) - min(y0s),
                    "line_length": len(line_text),
                    "num_cap_words": sum(1 for w in line_text.split() if w.isupper()),
                    "avg_word_len": sum(len(w) for w in line_text.split()) / len(line_text.split()) if line_text.split() else 0,
                    "page_num": page_num + 1
                }
                entries.append(entry)
    return entries

def vectorize(entry):
    return [
        entry["font_size"],
        entry["x0"],
        entry["y0"],
        entry["x1"],
        entry["y1"],
        entry["width"],
        entry["height"],
        entry["line_length"],
        entry["num_cap_words"],
        entry["avg_word_len"]
    ]
