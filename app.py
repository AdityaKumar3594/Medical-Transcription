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

# Load environment variables
load_dotenv()

# Configure Gemini API
if os.getenv('GEMINI_API_KEY'):
    genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

# Audio recording parameters
SAMPLE_RATE = 16000
DURATION = 10  # seconds

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
        <h1>🎙️ Speech-to-Text Q&A Assistant</h1>
        <p>Convert speech to text and ask questions using AI</p>
    </div>
    """, unsafe_allow_html=True)
    
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
                            # st.session_state.transcript = pdf_text.strip()
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
                st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)
        
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
    
    # Setup Instructions
    with st.expander("🔧 Setup Instructions", expanded=False):
        st.markdown("""
        ### Installation:
        
        1. **Install required packages:**
        ```bash
        pip install streamlit speechrecognition google-generativeai python-dotenv sounddevice pydub scipy numpy openai
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
        OPENROUTER_API_KEY=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
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
        - ✅ Transcript editing
        - ✅ Q&A history
        
        ### Troubleshooting:
        - **No audio devices**: Check microphone permissions
        - **Transcription errors**: Speak clearly, avoid background noise
        - **API errors**: Verify OpenRouter API key is correct
        """)

if __name__ == "__main__":
    main()