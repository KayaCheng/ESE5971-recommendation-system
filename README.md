## 1. Introduction

### Decisions to be impacted
This project is designed to support content-selection and recommendation decisions in a medical imaging learning or knowledge-support setting. The system is intended to improve the following decisions:

- **What content should be recommended next?**
  Given a user query, learning topic, or clinical interest, the system should recommend the most relevant document chunks, references, or learning materials.
- **Which source is most relevant for a given imaging concept?**
  For example, when the topic is Fourier transform, morphological operators, CT principles, or MRI basics, the system should rank the most useful materials higher.
- **How should content be organized for retrieval and downstream recommendation?**
  The project needs a reliable pipeline for extracting, cleaning, chunking, embedding, and ranking educational resources.
- **Which modeling approach gives the best tradeoff between quality, simplicity, and explainability?**
  This includes deciding whether a simple lexical baseline, a classical machine learning model, or an embedding-based retrieval pipeline is the best starting point.

### Business value
Although this is a course project, the practical value is strong:

- **Faster access to relevant medical imaging knowledge.** Users spend less time searching through long PDFs and more time reading the most relevant sections.
- **Better educational support.** Students can discover the right chapter, concept explanation, or technical background more efficiently.
- **Foundation for clinical decision support extensions.** A strong retrieval and recommendation pipeline can later support more advanced tools such as case-based recommendations or multimodal search.
- **Reusable data pipeline.** The preprocessing pipeline can be extended from PDF documents to web pages, lecture notes, videos, and other medical imaging resources.

### Why we care about this project
We care about this project because medical imaging is a knowledge-dense domain: textbooks, notes, and references are often long, technical, and difficult to search manually. A recommendation system can reduce friction in learning and information discovery. It also gives us a realistic course project that combines data engineering, information retrieval, machine learning, and system design in one end-to-end workflow.

## 2. Data and Data Preprocessing

### Current data scope
The current repository is centered on PDF-based educational material, especially a source PDF located under `data/raw/pdf/`. The current processing pipeline converts the raw PDF into:

- **Page-level cleaned records** saved as JSONL
- **Chunk-level records** saved as JSONL for downstream retrieval, embedding, and recommendation

### Current preprocessing pipeline
The implemented PDF pipeline has three main stages:

1. **Extraction**
   - Open the PDF with PyMuPDF
   - Extract raw text page by page
   - Store page number, source name, and raw text

2. **Cleaning**
   - Normalize line breaks
   - Replace PDF ligatures such as `ﬁ`, `ﬂ`
   - Remove standalone page numbers
   - Merge broken hyphenation across lines
   - Merge wrapped lines into paragraph-like text
   - Keep the main body pages only
   - Compute cleaned text length (`char_count`)

3. **Chunking**
   - Split cleaned pages into paragraph-like units
   - Detect likely titles and headings
   - Skip extremely short low-information pages
   - Build chunk records with chunk metadata
   - Link neighboring chunks using `prev_chunk_id` and `next_chunk_id`

### Data cleaning metrics we can report
For the project report, we should report simple and interpretable cleaning metrics before and after preprocessing. Recommended metrics include:

- Number of raw pages extracted
- Number of body pages retained after filtering
- Average characters per page before and after cleaning
- Number of pages removed for being empty or low-information
- Average chunk length in characters
- Estimated average chunk length in tokens
- Number of total chunks produced
- Percentage of chunks that fall within the target size range

### Outlier detection ideas
Outlier detection is important because noisy pages can hurt retrieval quality. We recommend labeling the following as outlier candidates:

- Pages with extremely low character count
- Pages containing mostly numbers or formatting artifacts
- Pages with repeated headers/footers
- Pages with very high symbol ratio or OCR noise
- Chunks that are too short to be meaningful
- Chunks that are too long and likely mix multiple concepts

Recommended quantitative thresholds for a first pass:

- **Low-information page:** fewer than 80 characters
- **Preferred chunk range:** roughly 600 to 1800 characters
- **Flagged chunk outlier:** below 300 characters or above 2200 characters

### Metrics to judge whether chunking is good
Chunk quality should not be judged by chunk length alone. We should track a mix of **structural metrics**, **retrieval metrics**, and **manual review metrics**.


ESE5971-recommendation-system/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   ├── pdf/
│   │   ├── video/
│   │   └── web/
│   └── processed/
│       ├── cleaned/
│       └── chunks/
├── src/
│   ├── pdf_processing/
│   ├── video_processing/
│   ├── web_processing/
│   ├── chunking/
│   └── utils/
├── notebooks/
└── tests/
