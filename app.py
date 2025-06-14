# app.py
import streamlit as st
import os
import tempfile
from pathlib import Path
import io
from docx import Document
import google.generativeai as genai
import time # For time.sleep during Gemini file processing

# --- Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("Gemini API Key not found. Please set the 'GEMINI_API_KEY' environment variable. "
             "If deploying on Render.com, add it under 'Environment Variables'.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

st.set_page_config(
    page_title="Video Content Extractor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Helper Function for Gemini Video Processing ---

def extract_text_with_gemini(video_file_path):
    """
    Attempts to extract textual information and a content summary from a video
    using the Gemini Pro Vision model.

    Important Note on Transcription:
    The current Gemini Pro Vision model primarily processes video frames for visual
    understanding and does not perform direct, robust, language-agnostic audio
    speech-to-text transcription.

    This function will analyze the video's visual content and context to generate
    a descriptive summary. For actual spoken word transcripts, a dedicated
    Automatic Speech Recognition (ASR) service (which is separate from the
    current Gemini API's primary video capabilities) would typically be required.
    """
    st.info("Initiating video analysis with Gemini Pro Vision. "
            "Please note: This model excels at understanding visual content. "
            "While it can infer context and potentially on-screen text, it does NOT "
            "perform direct audio-to-text transcription. The output will be a descriptive summary "
            "of the video's visual content based on frames.")

    uploaded_file_name = None
    try:
        # Step 1: Upload the video file to Gemini's backend for processing
        with st.spinner("Uploading video to Gemini API..."):
            video_file = genai.upload_file(video_file_path)
            uploaded_file_name = video_file.name

        # Step 2: Wait for the file to be processed by Gemini.
        processing_bar = st.progress(0, text="Processing video with Gemini API...")
        progress_percentage = 0
        while video_file.state.name == "PROCESSING":
            time.sleep(2) # Wait for 2 seconds before checking the status again
            video_file = genai.get_file(uploaded_file_name) # Refresh the file status
            progress_percentage = min(progress_percentage + 5, 99)
            processing_bar.progress(progress_percentage)

        if video_file.state.name == "FAILED":
            st.error("Failed to process video with Gemini API. Please try again or with a different video.")
            return "Error: Video processing failed."

        # Step 3: Define a detailed prompt for Gemini to describe the video content
        prompt = (
            "Please provide a comprehensive description of the content within this video. "
            "Focus on the main subjects, actions, settings, and any prominent text displayed. "
            "If there are clear visual cues suggesting spoken content or narrative, please infer "
            "and describe the general theme or topics being discussed based on visuals. "
            "Structure your response as several well-formatted paragraphs, suitable for a report or summary. "
            "The output should be language-agnostic in its descriptive nature, focusing on what is visually observable."
        )
        
            
        # Step 4: Generate content from the Gemini Pro Vision model
        model = genai.GenerativeModel('gemini-2.0-flash-thinking-exp-01-21')
        response = model.generate_content([prompt, video_file], stream=False)
        
        if response and hasattr(response, 'text'):
            return response.text
        else:
            st.error("Gemini API response did not contain expected text content.")
            return "Error: Gemini API did not return valid text."

    except Exception as e:
        st.error(f"An error occurred during Gemini API interaction: {e}")
        return "Error: Could not extract information using Gemini API."
    finally:
        if uploaded_file_name:
            try:
                genai.delete_file(uploaded_file_name)
                st.success("Temporary file successfully cleaned up from Gemini API.")
            except Exception as e:
                st.warning(f"Failed to delete uploaded file from Gemini API: {e}. "
                           "Please check your Gemini API usage or manually clear if possible.")

# --- Streamlit UI Layout ---

with st.sidebar:
    st.header("How to Use")
    st.info("""
    1.  **Upload your video file** using the "Upload a video file" button on the main page.
        (There is no client-side duration limit here; Gemini API may have its own limits.)
    2.  Once uploaded, click the **"Generate Content Summary"** button.
    3.  The app will then use the Google Gemini API to analyze your video's visual content.
    4.  The extracted descriptive text will be displayed in the main area. You can then
        copy it directly or download it as a Microsoft Word (.docx) document.
    """)

    st.header("About This App")
    st.markdown("""
    This application is designed to help you gain textual insights from your video content.
    It leverages the powerful **Google Gemini Pro Vision model** to analyze video frames
    and provide a comprehensive descriptive summary of the visual information.

    ---

    **Important Disclaimer on Audio Transcription:**
    While the app's purpose is to "extract text from video," it is crucial to understand
    that the Gemini Pro Vision model, as used here, primarily focuses on **visual content analysis**.
    It **does not perform direct audio-to-text transcription**. For accurate spoken word transcripts
    from audio tracks, a dedicated Automatic Speech Recognition (ASR) service would be required.
    This current implementation provides a descriptive summary of the video's visual elements,
    which can serve as a form of "extracted text" based on its visual narrative.
    """)

    st.header("Future Options & Features (Placeholder)")
    st.markdown("""
    * **True Audio Transcription:** Integration with a dedicated ASR service for spoken words.
    * **Smart Summarization:** Advanced AI-driven summarization and keyword extraction from content.
    * **Content Translation:** Ability to translate extracted content into various languages.
    * **Speaker Identification:** Diarization to identify different speakers in the video.
    * **Timestamped Events:** Generate summaries or transcripts with precise timestamps.
    * **More Input Formats:** Support for video URLs (e.g., YouTube links) in addition to file uploads.
    """)

st.title("🎥 Professional Video Content Extractor & Summarizer")
st.write("Upload your video to get a comprehensive textual summary of its visual content. "
         "This app leverages the advanced capabilities of the Google Gemini API to analyze your video "
         "and present insights in a clean, copyable, and downloadable format.")

uploaded_file = st.file_uploader(
    "Upload a video file (MP4, MOV, MKV, AVI, WEBM, etc.)",
    type=["mp4", "mov", "mkv", "avi", "webm"],
    help="Gemini API has its own internal limits for video size/duration. Please refer to Gemini API documentation for details."
)

transcript_display_area = st.empty()

if uploaded_file is not None:
    # Save the uploaded video to a temporary file on the server's disk
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        video_path = tmp_file.name

    st.video(video_path)

    if st.button("Generate Content Summary", type="primary", use_container_width=True):
        with st.spinner("Analyzing video content with Gemini API... This may take a moment based on video length and complexity."):
            extracted_text = extract_text_with_gemini(video_path)

        with transcript_display_area.container():
            st.markdown("---")
            st.subheader("Extracted Content Summary")
            st.markdown(extracted_text)

            st.text_area(
                "Copyable Summary",
                value=extracted_text,
                height=300,
                key="copy_summary_area",
                help="You can easily copy the entire summary from this text box using the built-in copy button."
            )

            doc = Document()
            doc.add_heading('Video Content Summary', level=1)
            for paragraph_text in extracted_text.split('\n\n'):
                doc.add_paragraph(paragraph_text)

            bio = io.BytesIO()
            doc.save(bio)
            bio.seek(0)

            st.download_button(
                label="Download as Word (DOCX)",
                data=bio.getvalue(),
                file_name="video_content_summary.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                help="Download the extracted content summary as a Microsoft Word document for offline use."
            )
    
    # Ensure the temporary video file is deleted from the local disk
    if os.path.exists(video_path):
        os.unlink(video_path)

