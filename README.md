
---

# 📘 Instagram Reel Classifier

Automatically classify Instagram reels into niche categories using captions + scraped web content + NLP similarity (TF-IDF + cosine similarity).

This project loads reel data (CSV/XLSX/ZIP), fetches textual metadata from reel URLs, processes text, and assigns each reel to a topic from a provided Topics.txt file.

---

## 🚀 Features

* **Auto-detection of file type**

  * Supports **CSV**, **Excel (.xlsx)**, **ZIP containing CSV/XLSX**, and **GZIP CSV**.
* **Robust CSV parser with auto-repair**

  * Fixes broken quotes, inconsistent delimiters, embedded newlines.
* **Web scraping** from Instagram reel URLs

  * Extracts text from meta tags, LD+JSON scripts, visible on-page text.
* **Text classification** using

  * TF-IDF vectorizer
  * Cosine similarity against predefined topics.
* **Multi-threaded URL fetching** for speed.
* **Automatic output saving** with `_with_predictions.csv`.

---

## 📁 Directory Structure

```
project/
│── classify_reels.py
│── README.md
│── data/
│     ├── dataset_instagram-reel.csv
│     ├── Topics.txt
│     └── output/
│          └── dataset_instagram-reel_classified.csv
```

---

## 🔧 Installation

### 1. Clone the repo

```bash
git clone <your-repo-url>
cd project
```

### 2. Create venv

```bash
python -m venv venv
```

### 3. Activate venv

**Windows**

```bash
venv\Scripts\activate
```

**Mac/Linux**

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install pandas requests beautifulsoup4 scikit-learn numpy openpyxl
```

---

## 🧠 How It Works

### 🔹 Step 1 — Load Topics

`Topics.txt` should contain one topic per line.

Example:

```
Wealth, Tax & Hidden Systems of the Rich
Saving Tricks and Tips
Safety Tips
Budgeting
...
```

### 🔹 Step 2 — Load Reels Dataset

The program automatically detects:

| File Type | Behavior                        |
| --------- | ------------------------------- |
| CSV       | Reads + repairs broken lines    |
| XLSX      | Reads using `pandas.read_excel` |
| ZIP       | Extracts first CSV/XLSX inside  |
| GZ        | Decompresses and parses         |

### 🔹 Step 3 — Fetch URL Content

Extracts:

* OG meta descriptions
* Twitter meta descriptions
* `<title>`
* LD+JSON scripts
* Visible page text

### 🔹 Step 4 — Combine Text

For classification, it merges:

```
caption + ownerUsername + ownerFullName + timestamp + fetched_text
```

### 🔹 Step 5 — Predict Topic

Uses TF-IDF (1–2 grams) + cosine similarity to match each reel to the most similar topic.

---

## ▶️ Usage Examples

### Basic

```bash
python classify_reels.py --input data/dataset_instagram-reel.csv --topics data/Topics.txt
```

### With Custom Output

```bash
python classify_reels.py \
  --input data/dataset_instagram-reel.csv \
  --topics data/Topics.txt \
  --output data/dataset_instagram-reel_classified.csv
```

### Show Sample Output

```bash
python classify_reels.py --input data/dataset_instagram-reel.csv --topics data/Topics.txt --show-sample
```

### Specify Encoding

```bash
python classify_reels.py -i data/reels.xlsx -t data/Topics.txt --encoding cp1252
```

---

## 📤 Output

The program generates a file:

```
dataset_instagram-reel_with_predictions.csv
```

with columns:

```
caption
ownerFullName
ownerUsername
url
timestamp
predicted_niche
```

---

## ⚠️ Troubleshooting

### ❗ Issue: "Only 35 rows out of 100 are classified"

**Cause:** Your file is NOT a CSV — it's an Excel `.xlsx` file with ZIP header (`PK\x03\x04`).
**Fix:** The new `read_dataset()` auto-detects and loads Excel/ZIP properly.

### ❗ "UnicodeDecodeError"

Use:

```bash
--encoding latin-1
```

### ❗ Predictions seem wrong?

Try adding more text sources in `build_combined_text()`.

### ❗ Requests blocked by Instagram

Add delay:

```bash
--delay-between-requests 0.5
```

or use a rotating proxy.

---

## 🛠 Advanced Options

| Flag                       | Description                |
| -------------------------- | -------------------------- |
| `--input`                  | Input CSV/XLSX/ZIP path    |
| `--topics`                 | Path to Topics.txt         |
| `--output`                 | Custom output CSV path     |
| `--encoding`               | Manual encoding override   |
| `--skip-bad-lines`         | Skip malformed CSV rows    |
| `--delay-between-requests` | Add delay between fetches  |
| `--show-sample`            | Print first 20 predictions |

---

## 📜 License

MIT License

---

