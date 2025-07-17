SMART MATCH RECRUITER  :


This is an AI-powered resume screening and matching web application built using Flask, BERT, SpaCy, and Transformers. It allows recruiters or job seekers to upload resumes and compare them against a job description using semantic similarity techniques, providing:

1.] Top resume matches

2.] Detailed similarity scores

3.] Section-wise feedback

4.] Personalized suggestions for improving resumes

🚀 Key Features
🔍 Semantic Matching: Uses BERT embeddings and cosine similarity to identify how closely a resume matches a job description.

📂 Multiple File Formats: Supports PDF, DOCX, and TXT resume uploads.

🧠 Section-Based Analysis: Evaluates key resume sections like Experience, Skills, Education, and Projects.

📈 Detailed Feedback: Suggests actionable improvements for weak sections.

📝 Explanations: Provides interpretable explanations for similarity scores.

🧰 Preprocessing Pipeline: Cleans and lemmatizes text using SpaCy before embedding.

🛠️ Tech Stack
Frontend: HTML (via Jinja templates)

Backend: Python, Flask

NLP:

Transformers (BERT embeddings)

SpaCy (text preprocessing)

Sentence-Transformers

Similarity Metric: Cosine Similarity (via scikit-learn)

File Parsing: docx2txt, PyPDF2

Hosting: Run locally via Flask development server

📂 Folder Structure
php
Copy code
├── app.py                    # Main Flask backend
├── templates/
│   └── matchresume.html      # Frontend page
├── uploads/                  # Folder for storing uploaded resumes
├── static/                   # (Optional) Static files like CSS or JS
└── requirements.txt          # Python dependencies
🧪 How It Works
User inputs a job description and uploads one or more resumes.

Text is extracted from resumes and preprocessed.

BERT embeddings are generated for the job description and resumes.

Cosine similarity scores are calculated.

The app:

Displays the top 5 matching resumes

Provides section-wise score explanations

Gives suggestions for improvement

✅ Future Enhancements
Add login/signup functionality

Support Google Drive or LinkedIn integration

Allow dynamic job roles and skill suggestions

Add voice-based input or output feedback

🧠 Ideal Use Cases
Job Seekers refining their resume for a specific role

Recruiters pre-screening large volumes of resumes

Career guidance tools or resume-building platforms

