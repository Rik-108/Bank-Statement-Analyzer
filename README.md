# Intelligent Bank Statement Analyzer 🏦📊

## 📌 Project Overview
**Intelligent Bank Statement Analyzer** is an enterprise-grade, full-stack financial application designed to automate the auditing of raw transaction data. Built with a **FastAPI** backend and integrated with **IBM Watsonx.ai**, the system ingests standard CSV bank statements and utilizes Large Language Models (LLMs) to perform complex financial reasoning. It moves beyond simple transaction categorization by offering real-time AI assessments for loan eligibility and sophisticated fraud detection.

## 🚀 Key Features
* **Automated Data Processing:** Utilizes `pandas` to instantly parse, clean, and summarize raw CSV bank statements into machine-readable financial contexts.
* **Enterprise AI Integration:** Connects securely to the **IBM Watsonx Prompt Lab** to execute deep financial analysis using advanced foundation models.
* **Multi-Faceted Intelligence:** Features dedicated endpoints for specific financial tasks, including generating general account summaries, red-flagging potential fraudulent activity, and scoring loan eligibility.
* **Modern Asynchronous UI:** Built with **Tailwind CSS** and custom "glassmorphism" styling, the frontend uses JavaScript `fetch` API for seamless, non-blocking AI communication and dynamic loading states.

---

## 🏗️ Architecture Details

### The Application Backend (`main.py`)
The server-side acts as a high-performance, asynchronous bridge between the user's sensitive data and the AI engine.
* **Framework:** Uses `FastAPI` to handle `UploadFile` requests, ensuring fast and secure processing of user-submitted CSVs.
* **Data Engineering:** Extracts the raw bytes into a virtual memory buffer (`io.StringIO`) and processes it via Pandas, preventing the need to save sensitive financial documents to a local disk.

### The Cognitive Engine (`watsonx_promptlab.py`)
The core intelligence relies on rigidly structured prompt engineering via the `ibm_watsonx_ai` SDK.
* **Model Parameters:** Employs precise text generation parameters, utilizing `"sample"` decoding with a temperature of `0.7` to balance analytical rigidity with the ability to spot nuanced fraud patterns.
* **Guardrails:** Uses defined `STOP_SEQUENCES` (e.g., `**Instructions**`, `**Analysis:**`) to force the LLM to adhere strictly to the requested output format and prevent hallucinated text generation.

---

## ⚖️ AI & Prompt Engineering Logic
Unlike a standard chatbot, this system uses highly specialized system prompts to force the AI into the role of a strict financial auditor:

1.  **Context Injection:** The summarized Pandas data is dynamically injected into the prompt template as the primary context block.
2.  **Task-Specific Reasoning:**
    * **Fraud Detection:** The AI is instructed to scan the summary specifically for "Red Flags" typical of money laundering or account takeover.
    * **Loan Assessment:** The AI cross-references the account's cash flow stability against standard lending criteria, explicitly outputting a binary "Yes/No" eligibility recommendation alongside its reasoning.

---

## 🛠️ Technology Stack
* **Backend Framework:** Python 3, FastAPI, Uvicorn
* **Data Processing:** Pandas
* **AI & LLM:** IBM Watsonx.ai (`ModelInference`)
* **Frontend UI:** HTML5, Tailwind CSS, Vanilla JavaScript
* **Environment Management:** `python-dotenv`

---

**Environment Variables Required (`.env`):**
* `WATSONX_REGION` (e.g., *us-south*)
* `WATSONX_API_KEY`
* `WATSONX_PROJECT_ID`

---

## 👥 Contributors
This project was developed as part of the **MBA in Data Science & Data Analytics (2024-2026)**.
* **Anik Basu** 
