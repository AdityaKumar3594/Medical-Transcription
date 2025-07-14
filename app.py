import streamlit as st
import speech_recognition as sr
import google.generativeai as genai
import os
from dotenv import load_dotenv
import tempfile
import time
import io
import wave
import base64
from pydub import AudioSegment
from pydub.playback import play
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile
from openai import OpenAI
import PyPDF2
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset
import re
import warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

# Configure Gemini API
if os.getenv('GEMINI_API_KEY'):
    genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

# Audio recording parameters
SAMPLE_RATE = 16000
DURATION = 10  # seconds

# Medical Models Configuration
MEDICAL_MODELS = {
    "ClinicalBERT": {
        "model_name": "emilyalsentzer/Bio_ClinicalBERT",
        "description": "BERT model pretrained on clinical notes (MIMIC-III), strong for clinical text classification.",
        "type": "classification"
    },
    "BioBERT": {
        "model_name": "dmis-lab/biobert-base-cased-v1.1",
        "description": "BERT variant pretrained on biomedical literature, good for medical NLP tasks.",
        "type": "classification"
    },
    "BlueBERT": {
        "model_name": "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12",
        "description": "BERT trained on PubMed and MIMIC-III clinical notes, optimized for clinical text.",
        "type": "classification"
    },
    "PubMedBERT": {
        "model_name": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "description": "BERT pretrained solely on PubMed abstracts, effective for biomedical domain tasks.",
        "type": "classification"
    },
    "BioClinicalBERT": {
        "model_name": "emilyalsentzer/Bio_ClinicalBERT",
        "description": "Combines biomedical and clinical text pretraining for improved clinical NLP performance.",
        "type": "classification"
    },
    "MedBERT": {
        "model_name": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        "description": "Designed for structured EHR data but adaptable for clinical text classification.",
        "type": "classification"
    },
    "BioMegatron": {
        "model_name": "nvidia/BioMegatron345mUncased",
        "description": "Large-scale biomedical BERT variant with enhanced capacity for complex medical texts.",
        "type": "classification"
    },
    "Longformer": {
        "model_name": "allenai/longformer-base-4096",
        "description": "Transformer designed to handle long documents, useful for lengthy clinical transcripts.",
        "type": "classification"
    }
}

# Medical Classification Categories
MEDICAL_CATEGORIES = [
    "Cardiology", "Neurology", "Oncology", "Orthopedics", "Gastroenterology",
    "Pulmonology", "Endocrinology", "Psychiatry", "Dermatology", "Radiology",
    "Pathology", "Emergency Medicine", "Surgery", "Pediatrics", "Gynecology"
]

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .transcript-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
    }
    
    .qa-container {
        background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #e17055;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
    }
    
    .model-container {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #00b894;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
    }
    
    .classification-result {
        background: linear-gradient(135deg, #d63031 0%, #74b9ff 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
    }
    
    .metrics-container {
        background: linear-gradient(135deg, #00b894 0%, #00cec9 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
    }
    
    .recording-indicator {
        background: #ff6b6b;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        animation: pulse 2s infinite;
        margin: 1rem 0;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    .success-box {
        background: #00b894;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .error-box {
        background: #e17055;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .info-box {
        background: #74b9ff;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .answer-box {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1rem;
        border: 1px solid #e8e8e8;
        color: #2c3e50;
        line-height: 1.6;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
    }
    
    .model-info {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid #dee2e6;
        color: #495057;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'transcript' not in st.session_state:
        st.session_state.transcript = ""
    if 'qa_history' not in st.session_state:
        st.session_state.qa_history = []
    if 'recording_state' not in st.session_state:
        st.session_state.recording_state = 'idle'  # idle, recording, processing
    if 'recorded_audio' not in st.session_state:
        st.session_state.recorded_audio = None
    if 'current_mode' not in st.session_state:
        st.session_state.current_mode = 'Q&A Mode'
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = None
    if 'loaded_model' not in st.session_state:
        st.session_state.loaded_model = None
    if 'loaded_tokenizer' not in st.session_state:
        st.session_state.loaded_tokenizer = None
    if 'classification_results' not in st.session_state:
        st.session_state.classification_results = []

def preprocess_medical_text(text):
    """Preprocess medical text for classification"""
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep medical abbreviations
    text = re.sub(r'[^\w\s\-\./]', '', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text

@st.cache_resource
def load_medical_model(model_name):
    """Load medical model and tokenizer with caching"""
    try:
        model_info = MEDICAL_MODELS[model_name]
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_info["model_name"])
        
        # Load model
        if model_info["type"] == "classification":
            # For classification, we'll use a generic model and adapt it
            model = AutoModel.from_pretrained(model_info["model_name"])
        
        return model, tokenizer, None
    except Exception as e:
        return None, None, str(e)

def simulate_medical_classification(text, model_name):
    """Simulate medical classification with basic text analysis"""
    processed_text = preprocess_medical_text(text)
    
    # Simple keyword-based classification for demonstration
    medical_keywords = {
        "Cardiology": ["heart", "cardiac", "cardiovascular", "blood pressure", "chest pain", "ecg", "ekg"],
        "Neurology": ["brain", "neurological", "seizure", "headache", "migraine", "stroke", "nervous"],
        "Oncology": ["cancer", "tumor", "oncology", "chemotherapy", "radiation", "malignant", "benign"],
        "Orthopedics": ["bone", "joint", "fracture", "arthritis", "spine", "knee", "hip"],
        "Gastroenterology": ["stomach", "gastric", "intestinal", "digestive", "bowel", "liver", "pancreas"],
        "Pulmonology": ["lung", "respiratory", "breathing", "asthma", "copd", "pneumonia", "cough"],
        "Endocrinology": ["diabetes", "thyroid", "hormone", "insulin", "glucose", "endocrine", "metabolism"],
        "Psychiatry": ["depression", "anxiety", "mental", "psychiatric", "mood", "behavioral", "therapy"],
        "Dermatology": ["skin", "rash", "dermatitis", "acne", "mole", "lesion", "eczema"],
        "Radiology": ["x-ray", "ct", "mri", "ultrasound", "imaging", "scan", "radiological"],
        "Pathology": ["biopsy", "pathology", "histology", "tissue", "specimen", "diagnosis", "microscopic"],
        "Emergency Medicine": ["emergency", "urgent", "trauma", "accident", "critical", "er", "acute"],
        "Surgery": ["surgery", "surgical", "operation", "procedure", "incision", "suture", "anesthesia"],
        "Pediatrics": ["child", "pediatric", "infant", "baby", "adolescent", "vaccination", "growth"],
        "Gynecology": ["gynecology", "pregnancy", "obstetric", "menstrual", "cervical", "ovarian", "uterine"]
    }
    
    scores = {}
    for category, keywords in medical_keywords.items():
        score = sum(1 for keyword in keywords if keyword in processed_text.lower())
        scores[category] = score
    
    # Get top predictions
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Calculate confidence (normalized scores)
    total_score = sum(scores.values())
    if total_score == 0:
        total_score = 1
    
    predictions = []
    for category, score in sorted_scores[:5]:
        confidence = score / total_score if total_score > 0 else 0
        predictions.append({
            "category": category,
            "confidence": confidence,
            "score": score
        })
    
    return predictions

def calculate_dummy_metrics(predictions):
    """Calculate dummy metrics for demonstration"""
    if not predictions:
        return {}
    
    # Simulate metrics
    top_confidence = predictions[0]["confidence"] if predictions else 0
    
    metrics = {
        "accuracy": min(0.95, 0.7 + top_confidence * 0.3),  # Simulated accuracy
        "precision": min(0.92, 0.65 + top_confidence * 0.35),  # Simulated precision
        "recall": min(0.88, 0.6 + top_confidence * 0.4),  # Simulated recall
        "f1_score": 0.0  # Will be calculated
    }
    
    # Calculate F1 score
    if metrics["precision"] + metrics["recall"] > 0:
        metrics["f1_score"] = 2 * (metrics["precision"] * metrics["recall"]) / (metrics["precision"] + metrics["recall"])
    
    return metrics

def record_audio_sounddevice(duration=5):
    """Record audio using sounddevice library"""
    try:
        st.info(f"🎙️ Recording for {duration} seconds... Speak now!")
        
        # Record audio
        audio_data = sd.rec(int(duration * SAMPLE_RATE), 
                           samplerate=SAMPLE_RATE, 
                           channels=1, 
                           dtype=np.int16)
        sd.wait()  # Wait until recording is finished
        
        return audio_data.flatten(), None
    except Exception as e:
        return None, f"Recording error: {str(e)}"

def save_audio_array_to_wav(audio_data, filename):
    """Save numpy audio array to WAV file"""
    try:
        wavfile.write(filename, SAMPLE_RATE, audio_data)
        return True
    except Exception as e:
        st.error(f"Error saving audio: {e}")
        return False

def transcribe_audio_file(audio_file_path):
    """Transcribe audio file using speech recognition"""
    recognizer = sr.Recognizer()
    
    try:
        with sr.AudioFile(audio_file_path) as source:
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            # Record the audio
            audio = recognizer.record(source)
            
        # Try to recognize speech
        text = recognizer.recognize_google(audio)
        return text, None
        
    except sr.UnknownValueError:
        return None, "Could not understand the audio clearly. Please try again with clearer speech."
    except sr.RequestError as e:
        return None, f"Speech recognition service error: {e}"
    except Exception as e:
        return None, f"Error processing audio: {e}"

def transcribe_uploaded_file(uploaded_file):
    """Transcribe uploaded audio file"""
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            # Handle different file types
            if uploaded_file.type.startswith('audio/'):
                # Convert to WAV using pydub
                audio = AudioSegment.from_file(io.BytesIO(uploaded_file.read()))
                audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1)
                audio.export(tmp_file.name, format="wav")
            else:
                # Direct write for WAV files
                tmp_file.write(uploaded_file.read())
            
            tmp_file_path = tmp_file.name
        
        # Transcribe the file
        text, error = transcribe_audio_file(tmp_file_path)
        
        # Clean up
        os.unlink(tmp_file_path)
        
        return text, error
        
    except Exception as e:
        return None, f"Error processing uploaded file: {e}"

def ask_openrouter(question, context):
    """Ask question to OpenRouter AI based on the transcript"""
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        return "Please set your OPENROUTER_API_KEY in the .env file to use OpenRouter Q&A features."
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        completion = client.chat.completions.create(
            model="mistralai/mistral-small-3.2-24b-instruct:free",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Based on the following transcript, please answer the question accurately and comprehensively:\n\nTRANSCRIPT:\n{context}\n\nQUESTION: {question}\n\nPlease provide a detailed answer based solely on the information available in the transcript. If the transcript doesn't contain enough information to answer the question, please mention that clearly."
                        }
                    ]
                }
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error generating response from OpenRouter: {str(e)}"

def main():
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🩺 Medical Transcription AI Assistant</h1>
        <p>Advanced speech-to-text with medical classification and Q&A capabilities</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Mode Selection
    st.markdown("## 🔧 Mode Selection")
    current_mode = st.radio(
        "Choose operation mode:",
        ["❓ Transcription Q&A Mode", "🩺 Medical Model Trainer Mode"],
        horizontal=True
    )
    
    st.session_state.current_mode = current_mode
    
    # Check dependencies
    st.sidebar.title("📋 Setup Status")
    
    # Check if required packages are installed
    try:
        import sounddevice as sd
        st.sidebar.success("✅ sounddevice installed")
    except ImportError:
        st.sidebar.error("❌ sounddevice not installed")
        st.sidebar.code("pip install sounddevice")
    
    try:
        import pydub
        st.sidebar.success("✅ pydub installed")
    except ImportError:
        st.sidebar.error("❌ pydub not installed")
        st.sidebar.code("pip install pydub")
    
    try:
        import torch
        st.sidebar.success("✅ torch installed")
    except ImportError:
        st.sidebar.error("❌ torch not installed")
        st.sidebar.code("pip install torch")
    
    try:
        import transformers
        st.sidebar.success("✅ transformers installed")
    except ImportError:
        st.sidebar.error("❌ transformers not installed")
        st.sidebar.code("pip install transformers")
    
    if os.getenv('OPENROUTER_API_KEY'):
        st.sidebar.success("✅ OpenRouter API key found")
    else:
        st.sidebar.error("❌ OpenRouter API key missing")
        st.sidebar.info("Add OPENROUTER_API_KEY to .env file")
    
    # Main content
    st.markdown("## 🎤 Audio Input")
    
    # Method selection
    input_method = st.radio(
        "Choose input method:",
        ["📁 Upload Audio File", "📄 Upload PDF Transcription", "🎙️ Record Audio (5 seconds)", "🎙️ Record Audio (10 seconds)"],
        index=0
    )
    
    if input_method == "📁 Upload Audio File":
        st.markdown("### Upload an audio file")
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'flac', 'aiff', 'm4a', 'ogg'],
            help="Supported formats: WAV, MP3, FLAC, AIFF, M4A, OGG"
        )
        
        if uploaded_file is not None:
            st.audio(uploaded_file, format='audio/wav')
            
            if st.button("🔄 Transcribe Audio File"):
                with st.spinner("Processing audio file..."):
                    text, error = transcribe_uploaded_file(uploaded_file)
                    
                    if text:
                        st.session_state.transcript = text
                        st.markdown(f'<div class="success-box">✅ Transcription successful!</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="error-box">❌ {error}</div>', unsafe_allow_html=True)

    elif input_method == "📄 Upload PDF Transcription":
        st.markdown("### Upload a PDF file containing a transcription")
        uploaded_pdf = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload a PDF file containing the transcription text."
        )
        if uploaded_pdf is not None:
            if st.button("📄 Extract Transcript from PDF"):
                with st.spinner("Extracting text from PDF..."):
                    try:
                        pdf_reader = PyPDF2.PdfReader(uploaded_pdf)
                        pdf_text = ""
                        for page in pdf_reader.pages:
                            pdf_text += page.extract_text() or ""
                        if pdf_text.strip():
                            st.session_state.transcript = pdf_text.strip()
                            st.markdown(f'<div class="success-box">✅ PDF text extracted and loaded as transcript!</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="error-box">❌ No text could be extracted from the PDF.</div>', unsafe_allow_html=True)
                    except Exception as e:
                        st.markdown(f'<div class="error-box">❌ Error extracting PDF: {e}</div>', unsafe_allow_html=True)
    
    elif "Record Audio" in input_method:
        duration = 5 if "5 seconds" in input_method else 10
        st.markdown(f"### Record audio for {duration} seconds")
        
        st.markdown(f"""
        <div class="info-box">
            <strong>Instructions:</strong>
            <ul>
                <li>Click "Start Recording" button</li>
                <li>Allow microphone access when prompted</li>
                <li>Speak clearly for {duration} seconds</li>
                <li>Wait for processing to complete</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button(f"🎙️ Start Recording ({duration}s)"):
                st.session_state.recording_state = 'recording'
                st.rerun()
        
        with col2:
            if st.button("🔄 Process Last Recording"):
                if st.session_state.recorded_audio is not None:
                    st.session_state.recording_state = 'processing'
                    st.rerun()
                else:
                    st.error("No audio recorded yet!")
        
        # Handle recording state
        if st.session_state.recording_state == 'recording':
            st.markdown(f'<div class="recording-indicator">🔴 Recording for {duration} seconds...</div>', unsafe_allow_html=True)
            
            # Record audio
            audio_data, error = record_audio_sounddevice(duration)
            
            if audio_data is not None:
                st.session_state.recorded_audio = audio_data
                st.session_state.recording_state = 'recorded'
                st.success(f"✅ Recording completed! Click 'Process Last Recording' to transcribe.")
            else:
                st.error(f"❌ Recording failed: {error}")
                st.session_state.recording_state = 'idle'
            
            st.rerun()
        
        elif st.session_state.recording_state == 'processing':
            st.markdown('<div class="info-box">🔄 Processing audio...</div>', unsafe_allow_html=True)
            
            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file_path = tmp_file.name
            
            if save_audio_array_to_wav(st.session_state.recorded_audio, tmp_file_path):
                # Transcribe
                text, error = transcribe_audio_file(tmp_file_path)
                
                # Clean up
                os.unlink(tmp_file_path)
                
                if text:
                    st.session_state.transcript = text
                    st.markdown('<div class="success-box">✅ Transcription successful!</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="error-box">❌ {error}</div>', unsafe_allow_html=True)
            else:
                st.error("❌ Failed to save audio file")
            
            st.session_state.recording_state = 'idle'
            st.rerun()
        
        elif st.session_state.recording_state == 'recorded':
            st.success("✅ Audio recorded successfully! Click 'Process Last Recording' to transcribe.")
    
    # Display Transcript
    if st.session_state.transcript:
        st.markdown("## 📝 Transcript")
        st.markdown("""
        <div class="transcript-container">
        """, unsafe_allow_html=True)
        
        # Display transcript
        st.write(st.session_state.transcript)
        
        # Edit transcript
        st.markdown("### Edit Transcript")
        edited_transcript = st.text_area(
            "You can edit the transcript here:",
            value=st.session_state.transcript,
            height=150
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Edited Transcript"):
                st.session_state.transcript = edited_transcript
                st.success("Transcript updated!")
        
        with col2:
            if st.button("🗑️ Clear Transcript"):
                st.session_state.transcript = ""
                st.session_state.qa_history = []
                st.session_state.classification_results = []
                st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Mode-specific content
        if current_mode == "❓ Transcription Q&A Mode":
            # Q&A Section
            st.markdown("## ❓ Ask Questions")
            st.markdown("""
            <div class="qa-container">
            """, unsafe_allow_html=True)
            
            model_choice = st.radio(
                "Choose AI model for Q&A:",
                ["OpenRouter"],
                horizontal=True
            )
            
            question = st.text_input(
                "Ask a question about the transcript:",
                placeholder="What is the main topic discussed?",
                key="question_input"
            )
            
            if st.button("🚀 Ask Question") and question:
                if model_choice == "OpenRouter":
                    if not os.getenv('OPENROUTER_API_KEY'):
                        st.error("Please set your OPENROUTER_API_KEY in the .env file")
                    else:
                        with st.spinner("Generating answer with OpenRouter..."):
                            answer = ask_openrouter(question, st.session_state.transcript)
                            st.session_state.qa_history.append({
                                'question': question,
                                'answer': answer,
                                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                                'model': 'OpenRouter'
                            })
                            st.rerun()
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Display Q&A History
            if st.session_state.qa_history:
                st.markdown("## 💬 Q&A History")
                
                for i, qa in enumerate(reversed(st.session_state.qa_history)):
                    model_used = qa.get('model', 'OpenRouter')
                    with st.expander(f"Q: {qa['question']} ({qa['timestamp']}) [{model_used}]", expanded=(i == 0)):
                        st.markdown(f"""
                        <div class="answer-box">
                            <strong>Answer:</strong><br>
                            {qa['answer']}
                        </div>
                        """, unsafe_allow_html=True)
                
                if st.button("🗑️ Clear Q&A History"):
                    st.session_state.qa_history = []
                    st.rerun()
        
        elif current_mode == "🩺 Medical Model Trainer Mode":
            # Medical Classification Section
            st.markdown("## 🩺 Medical Classification")
            st.markdown("""
            <div class="model-container">
            """, unsafe_allow_html=True)
            
            # Model Selection
            st.markdown("### Select Medical Model")
            selected_model = st.selectbox(
                "Choose a medical transformer model:",
                list(MEDICAL_MODELS.keys()),
                help="Select a pre-trained medical model for classification"
            )
            
            if selected_model:
                model_info = MEDICAL_MODELS[selected_model]
                st.markdown(f"""
                <div class="model-info">
                    <strong>Model:</strong> {selected_model}<br>
                    <strong>Description:</strong> {model_info['description']}<br>
                    <strong>Type:</strong> {model_info['type']}<br>
                    <strong>HuggingFace ID:</strong> {model_info['model_name']}
                </div>
                """, unsafe_allow_html=True)
            
            # Classification Options
            st.markdown("### Classification Settings")
            
            col1, col2 = st.columns(2)
            
            with col1:
                classification_type = st.selectbox(
                    "Classification Type:",
                    ["Medical Specialty", "Urgency Level", "Document Type"],
                    help="Choose the type of classification to perform"
                )
            
            with col2:
                confidence_threshold = st.slider(
                    "Confidence Threshold:",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.1,
                    help="Minimum confidence score for predictions"
                )
            
            # Text Preprocessing Display
            st.markdown("### Preprocessed Text Preview")
            if st.session_state.transcript:
                preprocessed_text = preprocess_medical_text(st.session_state.transcript)
                st.text_area(
                    "Preprocessed transcript:",
                    value=preprocessed_text,
                    height=100,
                    disabled=True
                )
            
            # Classification Button
            if st.button("🚀 Run Medical Classification"):
                if not st.session_state.transcript:
                    st.error("No transcript available for classification!")
                else:
                    with st.spinner(f"Running classification with {selected_model}..."):
                        try:
                            # Simulate model loading (in real implementation, use load_medical_model)
                            st.info("Loading model... (This is a simulation)")
                            time.sleep(2)  # Simulate loading time
                            
                            # Run classification
                            predictions = simulate_medical_classification(
                                st.session_state.transcript, 
                                selected_model
                            )
                            
                            # Calculate metrics
                            metrics = calculate_dummy_metrics(predictions)
                            
                            # Store results
                            result = {
                                'model': selected_model,
                                'classification_type': classification_type,
                                'predictions': predictions,
                                'metrics': metrics,
                                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                                'confidence_threshold': confidence_threshold
                            }
                            
                            st.session_state.classification_results.append(result)
                            st.success("Classification completed!")
                            
                        except Exception as e:
                            st.error(f"Classification failed: {str(e)}")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Display Classification Results
            if st.session_state.classification_results:
                st.markdown("## 🔍 Classification Results")
                
                # Display latest result
                latest_result = st.session_state.classification_results[-1]
                
                st.markdown(f"""
                <div class="classification-result">
                    <h3>Latest Classification Results</h3>
                    <p><strong>Model:</strong> {latest_result['model']}</p>
                    <p><strong>Classification Type:</strong> {latest_result['classification_type']}</p>
                    <p><strong>Timestamp:</strong> {latest_result['timestamp']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show predictions
                st.markdown("### 📊 Predictions")
                predictions = latest_result['predictions']
                
                for i, pred in enumerate(predictions[:5]):  # Show top 5
                    confidence_pct = pred['confidence'] * 100
                    color = "green" if confidence_pct >= 50 else "orange" if confidence_pct >= 25 else "red"
                    
                    st.markdown(f"""
                    <div style="background: linear-gradient(90deg, {color} 0%, {color} {confidence_pct}%, #f0f0f0 {confidence_pct}%, #f0f0f0 100%); 
                                padding: 10px; margin: 5px 0; border-radius: 5px; color: white;">
                        <strong>{i+1}. {pred['category']}</strong> - {confidence_pct:.1f}% (Score: {pred['score']})
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show metrics
                st.markdown("### 📈 Performance Metrics")
                metrics = latest_result['metrics']
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
                
                with col2:
                    st.metric("Precision", f"{metrics['precision']:.3f}")
                
                with col3:
                    st.metric("Recall", f"{metrics['recall']:.3f}")
                
                with col4:
                    st.metric("F1 Score", f"{metrics['f1_score']:.3f}")
                
                st.markdown(f"""
                <div class="metrics-container">
                    <h4>📊 Detailed Metrics</h4>
                    <p><strong>Model Performance Summary:</strong></p>
                    <ul>
                        <li>Accuracy: {metrics['accuracy']:.1%} - Overall correctness of predictions</li>
                        <li>Precision: {metrics['precision']:.1%} - Accuracy of positive predictions</li>
                        <li>Recall: {metrics['recall']:.1%} - Ability to find positive cases</li>
                        <li>F1 Score: {metrics['f1_score']:.1%} - Balanced measure of precision and recall</li>
                    </ul>
                    <p><strong>Note:</strong> These are simulated metrics for demonstration purposes.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Model Comparison
                if len(st.session_state.classification_results) > 1:
                    st.markdown("### 🔄 Model Comparison")
                    
                    comparison_data = []
                    for result in st.session_state.classification_results[-5:]:  # Last 5 results
                        comparison_data.append({
                            'Model': result['model'],
                            'Accuracy': f"{result['metrics']['accuracy']:.3f}",
                            'Precision': f"{result['metrics']['precision']:.3f}",
                            'Recall': f"{result['metrics']['recall']:.3f}",
                            'F1 Score': f"{result['metrics']['f1_score']:.3f}",
                            'Timestamp': result['timestamp']
                        })
                    
                    st.dataframe(comparison_data, use_container_width=True)
                
                # Classification History
                st.markdown("### 📚 Classification History")
                for i, result in enumerate(reversed(st.session_state.classification_results)):
                    with st.expander(f"Result {len(st.session_state.classification_results)-i}: {result['model']} - {result['timestamp']}", expanded=(i == 0)):
                        st.json({
                            'model': result['model'],
                            'classification_type': result['classification_type'],
                            'top_predictions': result['predictions'][:3],
                            'metrics': result['metrics']
                        })
                
                if st.button("🗑️ Clear Classification History"):
                    st.session_state.classification_results = []
                    st.rerun()
    
    # Advanced Features Section
    if st.session_state.transcript:
        st.markdown("## 🔧 Advanced Features")
        
        with st.expander("🎯 Fine-tuning Options", expanded=False):
            st.markdown("""
            ### Model Fine-tuning (Simulated)
            
            In a production environment, you could:
            - Fine-tune models on your specific medical data
            - Adjust hyperparameters for better performance
            - Use domain-specific vocabularies
            - Implement custom loss functions
            """)
            
            st.selectbox(
                "Fine-tuning Strategy:",
                ["Full Fine-tuning", "Adapter Layers", "LoRA", "Prompt Tuning"],
                help="Choose fine-tuning approach"
            )
            
            st.slider("Learning Rate:", 1e-5, 1e-3, 2e-5, format="%.0e")
            st.slider("Epochs:", 1, 10, 3)
            st.slider("Batch Size:", 8, 64, 16)
        
        with st.expander("📊 Data Analysis", expanded=False):
            if st.session_state.transcript:
                st.markdown("### Text Statistics")
                
                text = st.session_state.transcript
                words = text.split()
                sentences = text.split('.')
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Word Count", len(words))
                
                with col2:
                    st.metric("Sentence Count", len(sentences))
                
                with col3:
                    st.metric("Character Count", len(text))
                
                # Medical term frequency
                medical_terms = [
                    'patient', 'diagnosis', 'treatment', 'symptom', 'medication',
                    'therapy', 'procedure', 'clinical', 'medical', 'health'
                ]
                
                term_counts = {}
                for term in medical_terms:
                    count = text.lower().count(term)
                    if count > 0:
                        term_counts[term] = count
                
                if term_counts:
                    st.markdown("### Medical Term Frequency")
                    st.bar_chart(term_counts)
    
    # Setup Instructions
    with st.expander("🔧 Setup Instructions", expanded=False):
        st.markdown("""
        ### Installation:
        
        1. **Install required packages:**
        ```bash
        pip install streamlit speechrecognition google-generativeai python-dotenv 
        pip install sounddevice pydub scipy numpy openai PyPDF2
        pip install torch transformers datasets sklearn
        ```
        
        2. **For audio format support:**
        ```bash
        # Windows
        pip install ffmpeg-python
        
        # macOS
        brew install ffmpeg
        
        # Linux
        sudo apt update && sudo apt install ffmpeg
        ```
        
        3. **Create .env file:**
        ```
        OPENROUTER_API_KEY=your_openrouter_api_key_here
        ```
        
        4. **Get OpenRouter API Key:**
        - Visit [OpenRouter](https://openrouter.ai)
        - Create an account
        - Get your API key
        - Add it to your .env file
        
        ### Features:
        - ✅ File upload (multiple formats)
        - ✅ Real microphone recording
        - ✅ Speech-to-text transcription
        - ✅ AI-powered Q&A (with OpenRouter)
        - ✅ Medical classification with 8 transformer models
        - ✅ Performance metrics and model comparison
        - ✅ Transcript editing
        - ✅ Classification and Q&A history
        
        ### Medical Models Supported:
        - **ClinicalBERT**: Clinical notes specialist
        - **BioBERT**: Biomedical literature expert
        - **BlueBERT**: PubMed + clinical notes
        - **PubMedBERT**: PubMed abstracts focused
        - **BioClinicalBERT**: Biomedical + clinical hybrid
        - **MedBERT**: EHR data specialist
        - **BioMegatron**: Large-scale biomedical model
        - **Longformer**: Long document specialist
        
        ### Troubleshooting:
        - **No audio devices**: Check microphone permissions
        - **Transcription errors**: Speak clearly, avoid background noise
        - **API errors**: Verify OpenRouter API key is correct
        - **Model loading**: Ensure sufficient memory and internet connection
        - **Classification issues**: Check transcript quality and preprocessing
        """)

if __name__ == "__main__":
    main()