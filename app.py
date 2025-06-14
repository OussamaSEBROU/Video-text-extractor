import streamlit as st
import google.generativeai as genai
from docx import Document
from io import BytesIO
import os
import tempfile
import time # For polling Gemini file status

# REMOVE: from pydub import AudioSegment
# ADD: ffmpeg-python import
import ffmpeg # This is the new module for audio extraction

# Import pydub for audio extraction
from pydub import AudioSegment
# pydub requires ffmpeg or libav to be installed on the system.
# We'll handle this in the Dockerfile.

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="AI Video Transcriber",
    page_icon="🎥",
    layout="wide", # Use a wide layout for better content display
    initial_sidebar_state="expanded"
)

# --- Gemini API Key Configuration ---
# This attempts to get the API key from environment variables (for Render)
# or Streamlit secrets (for local development).
try:
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        gemini_api_key = st.secrets["GEMINI_API_KEY"]
    
    genai.configure(api_key=gemini_api_key)
    
    # Initialize the Gemini model for audio processing
    # gemini-1.5-flash-latest is generally good for speed and cost.
    # For very long or complex audio, consider gemini-1.5-pro-latest if your quota allows.
    model = genai.GenerativeModel('gemini-2.0-flash')

except KeyError:
    st.error("Gemini API key not found. Please set `GOOGLE_API_KEY` in your environment variables (for Render) or in `.streamlit/secrets.toml` (for local development).")
    st.stop()
except Exception as e:
    st.error(f"Error configuring Gemini API: {e}")
    st.stop()

# --- Helper Functions ---

def extract_audio_from_video(video_path):
    """
    Extracts audio from a video file using ffmpeg-python and saves it temporarily as MP3.
    Requires ffmpeg to be installed on the system (handled by Dockerfile).
    """
    try:
        # Create a temporary file path for the audio output
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio_file:
            temp_audio_path = temp_audio_file.name
        
        st.info(f"Loading video: {video_path}")
        
        # Use ffmpeg-python to extract audio.
        # input(video_path) creates an ffmpeg input stream from your video file.
        # output(temp_audio_path, acodec='libmp3lame') configures the output to an MP3 file
        # with the 'libmp3lame' audio codec.
        # .run() executes the ffmpeg command.
        # overwrite_output=True is good for temporary files to ensure they can be overwritten.
        # capture_stderr=True helps in debugging if ffmpeg throws an error.
        ffmpeg.input(video_path).output(temp_audio_path, acodec='libmp3lame').run(overwrite_output=True, capture_stderr=True)
        
        return temp_audio_path
    except ffmpeg.Error as e:
        # Catch specific ffmpeg errors to provide more context
        st.error(f"Error extracting audio with ffmpeg-python. Check ffmpeg logs: {e.stderr.decode()}")
        st.info("Ensure ffmpeg is installed and accessible in the environment.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during audio extraction: {e}")
        return None

def transcribe_audio_with_gemini(audio_path):
    """
    Transcribes an audio file using Google Gemini API.
    Uploads the file, polls for processing status, and then transcribes.
    """
    try:
        st.info("Uploading audio for transcription to Google Gemini...")
        audio_file = genai.upload_file(path=audio_path)
        
        with st.spinner(f"Processing audio file '{audio_file.display_name}' with Gemini... This might take a moment."):
            while audio_file.state.name == "PROCESSING":
                time.sleep(2)
                audio_file = genai.get_file(audio_file.name)

        if audio_file.state.name == "FAILED":
            st.error(f"Audio file processing failed on Gemini's side for: {audio_file.display_name}. Please try again or check the file format.")
            genai.delete_file(audio_file.name)
            return None

        st.info("Transcribing audio content... This can take longer for larger files.")
        response = model.generate_content(["Transcribe the following audio, providing only the spoken text:", audio_file])
        
        genai.delete_file(audio_file.name)
        
        return response.text if response.text else "No transcription found."
    except genai.types.BlockedPromptException as e:
        st.error(f"Transcription blocked: {e.response.prompt_feedback.block_reason_message}. Please adjust content or retry.")
        return None
    except Exception as e:
        st.error(f"An error occurred during Gemini transcription: {e}")
        st.warning("Please ensure your video is not excessively long, as there are API usage limits.")
        try:
            if 'audio_file' in locals() and audio_file.name:
                genai.delete_file(audio_file.name)
        except Exception as cleanup_e:
            st.warning(f"Failed to clean up Gemini file after error: {cleanup_e}")
        return None

def create_word_document(text):
    """Creates a simple Word document (BytesIO object) from text."""
    document = Document()
    document.add_paragraph(text)
    
    bio = BytesIO()
    document.save(bio)
    bio.seek(0)
    return bio

# --- Streamlit UI Layout ---

# Sidebar Content
with st.sidebar:
    st.header("About This App ℹ️")
    st.markdown(
        """
        This application harnesses the power of **Google Gemini AI**
        to convert the spoken words from your video files into written text (transcription).
        It's designed for efficiency and ease of use, providing a quick way
        to get textual content from your video recordings.
        """
    )
    
    st.header("How to Use 🚀")
    st.markdown(
        """
        1.  **Upload Your Video:** Use the file uploader in the main area to select your video file.
            (Supported formats: MP4, MOV, AVI, MKV).
        2.  **Initiate Transcription:** Click the "Transcribe Video" button. The app will first
            extract the audio, then send it to Google Gemini for transcription.
            This process can take some time depending on video length and file size.
        3.  **Review & Download:** Once the transcription is complete, the extracted text
            will be displayed below. You can easily copy the text or download it
            as a Microsoft Word (`.docx`) document.
        """
    )
    
    st.header("Help & Support ❓")
    st.markdown(
        """
        *   **Supported Video Formats:** MP4, MOV, AVI, MKV.
        *   **Audio Clarity:** For best results, ensure the audio in your video is clear with minimal background noise.
        *   **Processing Time:** Longer or higher-quality videos will take more time to process.
            Please be patient, especially with large files.
        *   **Troubleshooting:**
            *   If transcription fails, try a shorter video.
            *   Verify your `GOOGLE_API_KEY` is correctly set up.
            *   Check the app's logs on Render.com for more detailed error messages.
        *   **Limitations:** This app currently only extracts spoken dialogue. On-screen text (OCR) is not supported in this version.
        """
    )
    
    st.header("Developed By")
    st.markdown("Your Name/Company Name")
    st.markdown("[Your Website/GitHub/LinkedIn Link (Optional)]")
    st.markdown("---")
    st.info("Powered by Google Gemini & Streamlit.")

# Main Area Content
st.title("AI Video to Text Transcriber 🎥✍️")
st.markdown("""
Upload your video file here. Our intelligent system will extract the audio and provide
a comprehensive text transcription of the spoken content.
""")

uploaded_file = st.file_uploader(
    "Upload a video file to transcribe",
    type=["mp4", "mov", "avi", "mkv"],
    accept_multiple_files=False,
    help="Select a video file (MP4, MOV, AVI, MKV). File size limits may apply based on deployment platform."
)

transcribed_text = ""
video_path = None # Initialize to None

if uploaded_file is not None:
    st.subheader("Uploaded Video Preview:")
    st.video(uploaded_file, format="video/mp4", start_time=0)
    
    st.info("Saving video to a temporary location...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_video_file:
        tmp_video_file.write(uploaded_file.read())
        video_path = tmp_video_file.name
    st.success(f"Video '{uploaded_file.name}' uploaded successfully and ready for processing!")

    if st.button("Transcribe Video", help="Click to start the transcription process."):
        if video_path:
            with st.spinner("Step 1/2: Extracting audio from video..."):
                audio_file_path = extract_audio_from_video(video_path)

            if audio_file_path:
                transcribed_text = transcribe_audio_with_gemini(audio_file_path)
                
                try:
                    os.remove(audio_file_path)
                    st.success("Temporary audio file cleaned up.")
                except OSError as e:
                    st.warning(f"Could not remove temporary audio file: {e}")
            else:
                st.error("Audio extraction failed. Cannot proceed with transcription.")
        else:
            st.error("No video file found to transcribe. Please upload a video first.")
    
    if video_path and os.path.exists(video_path):
        try:
            os.remove(video_path)
            st.info("Temporary video file cleaned up.")
        except OSError as e:
            st.warning(f"Could not remove temporary video file: {e}")

if transcribed_text:
    st.subheader("🎉 Extracted Text (Transcription):")
    st.text_area(
        "Transcription Output",
        transcribed_text,
        height=400,
        help="This is the automatically transcribed text from your video's audio. You can select and copy it."
    )
    
    col1, col2 = st.columns([0.25, 0.75])
    
    word_doc_bytes = create_word_document(transcribed_text)
    with col1:
        st.download_button(
            label="📄 Download as Word (.docx)",
            data=word_doc_bytes,
            file_name="video_transcription.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            help="Download the transcribed text as a Microsoft Word document."
        )

    with col2:
        st.info("To copy the text, simply highlight it in the 'Transcription Output' box and press Ctrl+C (Cmd+C).")

elif uploaded_file is None:
    st.info("Upload your video file above and click 'Transcribe Video' to begin the process.")