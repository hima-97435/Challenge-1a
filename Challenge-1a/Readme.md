# 🧠 Adobe Hackathon 2025 – Challenge 1a: PDF Structure Extraction

## 📌 Problem Statement
The task is to automatically extract structured outlines from PDF documents and generate a JSON summary with hierarchical headings (`H1`, `H2`, `H3`) as per the given schema.

## ✅ Solution Overview
This solution uses a **machine learning-based approach** with a `RandomForestClassifier` trained on manually labeled heading data extracted from various PDFs.

The pipeline includes:
- Text and layout feature extraction from PDFs
- Custom feature engineering for ML
- Heading classification (H1/H2/H3/Sub/None)
- Structured JSON generation conforming to the schema

## 🧪 Features Used
- Font size, width, height, position (x/y)
- Text length, average word length
- Page number, line spacing
- [Augmentation] applied to under-represented classes (like H3)

## 🧰 Tech Stack
- Python 3.10 (via `slim` base)
- `PyMuPDF` for PDF parsing
- `Scikit-learn` for ML
- `Docker` for containerization

## 🐳 Dockerized Execution

### 🔧 Build the image
```bash
docker build --platform linux/amd64 -t pdf-processor .
