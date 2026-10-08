# 📈 PSX Investor Intelligence RAG

Welcome to the **PSX Investor Intelligence** app! Yeh ek source-grounded Retrieval-Augmented Generation (RAG) engine hai jo Pakistan Stock Exchange (PSX) ki companies ke financial documents analyze karne ke liye banaya gaya hai.

## 🚀 Yeh App Kya Karti Hai?
Yeh AI hawa mein teer nahi chalati (no hallucinations). Yeh seedha aapke diye gaye Google Drive folder se pichle 5 saal ki Annual aur Quarterly reports (PDFs, Excels) parhti hai. Jab aap koi sawal poochte hain, to yeh sirf unhi reports se saboot (evidence) nikal kar jawab deti hai aur sath exact references (citations) bhi batati hai.

## 📂 Repository ki Files
* `app.py`: App ka main dimaagh (Python script jisme extraction, indexing aur AI logic hai).
* `requirements.txt`: Python ke zaroori tools ki list.
* `packages.txt`: Linux system files (Tesseract OCR) jo Streamlit Cloud ko scanned PDF parhne ke liye chahiye.
* `README.md`: Yeh hidayat nama (Instruction file).

## ☁️ Streamlit Cloud Par Deploy Karne Ka Tariqa
1. [share.streamlit.io](https://share.streamlit.io/) par jayen aur apne GitHub account se login karein.
2. **Create app** -> **Yup, I have an app** par click karein.
3. Is repository ko select karein, branch mein `main`, aur file mein `app.py` likha rehne dein.
4. **Deploy** dabane se pehle **Advanced settings...** par click karein.
5. **Secrets** ke box mein apni Gemini API Key is tarah daalein:
   ```toml
   GEMINI_API_KEY = "Yahan_Apni_Asli_API_Key_Dalein"
