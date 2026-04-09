# NEXORA — Placement Resources Directory

This folder contains documents that feed the RAG (Retrieval-Augmented Generation) pipeline.

## Expected Structure

```
placement_resources/
├── job_descriptions/     # JD files (.txt, .md, .pdf)
├── dsa/                  # DSA study materials
├── hr_guides/            # HR interview prep guides
├── company_profiles/     # Company info and hiring processes
└── learning_resources/   # Tutorials, course recommendations
```

## Supported Formats
- `.txt`, `.md`, `.csv`, `.json` — text files
- `.pdf` — parsed via PyMuPDF

## Category Auto-Detection
Files are auto-categorized based on filename/path keywords:
- `jd`, `job_description` → `job_descriptions`
- `dsa`, `algorithm` → `dsa`
- `hr`, `behavioral` → `hr_guides`
- `company`, `profile` → `company_profiles`
- `resource`, `learn` → `learning_resources`

## Running Ingestion
```bash
python -m rag.ingest
```

> **Note:** If no documents are found, the pipeline auto-creates sample data for bootstrapping.
