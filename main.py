from flask import Flask, request, render_template
import os
import docx2txt
import PyPDF2
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch
import logging
from collections import Counter

# Initialize Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load SpaCy model for NLP tasks
nlp = spacy.load('en_core_web_sm')

# Load BERT model and tokenizer (for better sentence embedding)
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Define sections to analyze (for explanation purposes)
SECTIONS = ['experience', 'skills', 'education', 'projects']

def extract_text_from_pdf(file_path):
    # Extracts text from a PDF file
    text = ""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        logging.error(f"Error reading PDF file {file_path}: {e}")
    return text

def extract_text_from_docx(file_path):
    # Extracts text from a DOCX file
    try:
        return docx2txt.process(file_path)
    except Exception as e:
        logging.error(f"Error reading DOCX file {file_path}: {e}")
        return ""

def extract_text_from_txt(file_path):
    # Extracts text from a TXT file
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        logging.error(f"Error reading TXT file {file_path}: {e}")
        return ""

def extract_text(file_path):
    # Extracts text based on file extension
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    elif file_path.endswith('.txt'):
        return extract_text_from_txt(file_path)
    else:
        logging.warning(f"Unsupported file format for {file_path}")
        return ""

def preprocess_text(text):
    # Preprocesses text by tokenizing, lemmatizing, and removing stop words
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

def get_bert_embeddings(text):
    # Gets the BERT embeddings for a given text
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()  # Return the average embedding

def generate_section_embeddings(text):
    # Extract section-based embeddings
    sections = {section: "" for section in SECTIONS}
    for section in SECTIONS:
        if section in text.lower():
            sections[section] = " ".join([sentence.text for sentence in nlp(text).sents if section in sentence.text.lower()])
    
    section_embeddings = {section: get_bert_embeddings(preprocess_text(content)) for section, content in sections.items() if content}
    return section_embeddings

def generate_explanations(job_description, resume):
    # Generate explanations for similarity score based on sections
    explanations = []
    job_sections = generate_section_embeddings(job_description)
    resume_sections = generate_section_embeddings(resume)
    
    for section, job_embedding in job_sections.items():
        if section in resume_sections:
            resume_embedding = resume_sections[section]
            score = cosine_similarity([job_embedding], [resume_embedding])[0][0]
            explanations.append(f"{section.capitalize()} section similarity: {round(score, 2)}")
    
    return explanations if explanations else ["No specific sections matched significantly."]

def generate_suggestions(job_description, resume):
    # Generate suggestions based on the comparison of sections in the job description and resume
    suggestions = []
    job_sections = generate_section_embeddings(job_description)
    resume_sections = generate_section_embeddings(resume)
    
    for section in SECTIONS:
        if section in job_sections:
            job_embedding = job_sections[section]
            if section in resume_sections:
                resume_embedding = resume_sections[section]
                score = cosine_similarity([job_embedding], [resume_embedding])[0][0]
                
                # Suggest improvements based on section similarities
                if score < 0.5:  # Arbitrary threshold for poor matching
                    suggestions.append(f"Consider enhancing your {section} section. The similarity score is low. "
                                       f"Try to focus more on the key skills and experience for this role.")
                else:
                    suggestions.append(f"Your {section} section is a strong match, but could be improved further. "
                                       f"Consider elaborating on specific experiences or skills that align with the job.")
            else:
                suggestions.append(f"The {section} section is missing in your resume. Consider adding relevant details, "
                                   f"especially about your experience, skills, or projects in this area.")
        else:
            suggestions.append(f"The {section} section doesn't appear to be well-represented in the job description. "
                               f"Consider highlighting more relevant experience or skills in this section.")
    
    # Add general feedback if sections are good but can be enhanced
    suggestions.append("Additionally, consider customizing your resume further to highlight achievements and measurable outcomes.")
    suggestions.append("Ensure that your resume clearly demonstrates how your skills align with the job requirements, "
                       "particularly focusing on experience and projects that directly relate to the role.")
    suggestions.append("Make sure that any technical skills mentioned are up-to-date and specific to the tools or technologies used in the job.")

    return suggestions

def save_resume_file(resume_file):
    """
    Saves the uploaded resume file to the uploads folder.
    """
    filename = os.path.join(app.config['UPLOAD_FOLDER'], resume_file.filename)
    resume_file.save(filename)
    return filename

@app.route("/")
def matchresume():
    # Renders the resume matching homepage
    return render_template('matchresume.html')

@app.route('/matcher', methods=['POST'])
def matcher():
    if request.method == 'POST':
        job_description = request.form.get('job_description', '')
        resume_files = request.files.getlist('resumes')

        if not resume_files or not job_description:
            logging.warning("No resumes or job description provided.")
            return render_template('matchresume.html', message="Please upload resumes and enter a job description.")

        logging.info("Processing resumes and job description for matching.")
        
        # Preprocess the job description
        job_description_processed = preprocess_text(job_description)
        
        # Process resumes and extract preprocessed texts
        resumes = [extract_text(save_resume_file(resume)) for resume in resume_files]
        resume_texts = [preprocess_text(resume) for resume in resumes]

        # Generate embeddings using BERT (BERT is used for matching)
        job_embedding = get_bert_embeddings(job_description_processed)
        resume_embeddings = [get_bert_embeddings(resume) for resume in resume_texts]

        # Calculate cosine similarities
        similarities = cosine_similarity([job_embedding], resume_embeddings)[0]

        # Sort and get top 5 matching resumes
        top_indices = similarities.argsort()[-5:][::-1]
        top_resumes = [resume_files[i].filename for i in top_indices]
        similarity_scores = [round(similarities[i], 2) for i in top_indices]

        # Check if any significant matches were found (e.g., threshold for a good match)
        if all(score < 0.5 for score in similarity_scores):  # If all scores are below 0.5
            return render_template('matchresume.html', message="No significant matches found. Please consider improving your resume based on the suggestions provided.",
                                   top_resumes=top_resumes, similarity_scores=similarity_scores,
                                   resume_suggestions=resume_suggestions, score_explanations=score_explanations)

        # Generate improvement suggestions and explanations for each top resume
        resume_suggestions = []
        score_explanations = []
        for index in top_indices:
            resume_text = resumes[index]
            suggestions = generate_suggestions(job_description, resume_text)
            resume_suggestions.append(suggestions)
            explanations = generate_explanations(job_description, resume_text)
            score_explanations.append(explanations)

        logging.info("Matching complete. Displaying top resumes.")
        
        return render_template('matchresume.html', message="Top matching resumes:", top_resumes=top_resumes, 
                               similarity_scores=similarity_scores, resume_suggestions=resume_suggestions,
                               score_explanations=score_explanations)

    return render_template('matchresume.html')

def ensure_upload_folder():
    # Ensures the upload folder exists
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        logging.info(f"Creating upload folder at {app.config['UPLOAD_FOLDER']}")
        os.makedirs(app.config['UPLOAD_FOLDER'])

if __name__ == '__main__':
    ensure_upload_folder()
    logging.info("Starting Flask application.")
    app.run(debug=True)
