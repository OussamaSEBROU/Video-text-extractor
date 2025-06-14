



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
    Extracts comprehensive visual content and on-screen text from a video
    using the Gemini Pro Vision model. This function aims for a "transcription-like"
    output based purely on visual observation.

    Important Note on Transcription:
    This model excels at understanding visual content and reading on-screen text.
    It **does NOT perform direct audio-to-text transcription**. The output will be
    a detailed descriptive summary of the video's visual elements and any text
    appearing in frames, structured to resemble a continuous narrative or report.
    For true spoken word transcription, a dedicated ASR service is required.
    """
    st.info("Initiating video analysis with Gemini Pro Vision for visual content extraction. "
            "Please note: This process focuses on what is *visually observable* in the video, "
            "including any on-screen text, and infers narrative from visual cues. "
            "It does NOT perform audio-to-text transcription.")

    uploaded_file_name = None
    try:
        with st.spinner("Uploading video to Gemini API..."):
            video_file = genai.upload_file(video_file_path)
            uploaded_file_name = video_file.name

        processing_bar = st.progress(0, text="Processing video with Gemini API...")
        progress_percentage = 0
        while video_file.state.name == "PROCESSING":
            time.sleep(2)
            video_file = genai.get_file(uploaded_file_name)
            progress_percentage = min(progress_percentage + 5, 99)
            processing_bar.progress(progress_percentage)

        if video_file.state.name == "FAILED":
            st.error("Failed to process video with Gemini API. Please try again or with a different video.")
            return "Error: Video processing failed."

        # --- Refined Prompt for Direct Content Extraction ---
        prompt = (
            "You are a highly precise content extraction AI. Your task is to extract the text from the video without change anything thing in the content"
" the language of extracted text should be same of the video .. be carefuly on that"

            "Present the extracted content in clean, well-formatted paragraphs, ordering events chronologically as they appear in the video. "
           
        )
        
        # Using gemini-pro-vision, as it's the multimodal model for video input
        model = genai.GenerativeModel('gemini-2.0-flash')
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
        (Gemini API has its own internal limits for video size/duration, typically up to 2 minutes for direct file uploads.)
    2.  Once uploaded, click the **"Generate Content Summary"** button.
    3.  The app will then use the Google Gemini API to analyze your video's **visual content**.
    4.  The extracted descriptive text (visual "transcription") will be displayed in the main area.
        You can then copy it directly or download it as a Microsoft Word (.docx) document.
    """)

    st.header("About This App")
    st.markdown("""
    This application helps you get detailed textual insights from your video content by leveraging the
    **Google Gemini Pro Vision model**. It analyzes video frames to provide a comprehensive and
    structured summary of **all visually observable information**, including on-screen text,
    actions, and visual narratives.

    ---

    **CRITICAL CLARIFICATION: Visual vs. Audio Transcription**

    It is paramount to understand that this app, using `gemini-pro-vision`, performs **visual content extraction and on-screen text transcription only.**

    * **✅ What it DOES:** Provides a highly detailed description of everything visible in the video frames. It will capture and output any text that appears on screen (like captions, titles, or presentation slides). It will describe visual cues that might suggest dialogue (e.g., "Person A is seen speaking to Person B").
    * **❌ What it DOES NOT do:** It **does NOT "listen" to the audio track** of your video. Therefore, it cannot transcribe spoken words, distinguish between multiple speakers based on their voices, or generate a verbatim transcript of a conversation. For that, a dedicated **Automatic Speech Recognition (ASR)** service is required, which is a different technology.

    The output you receive will be a "visual transcription" – a detailed, paragraph-ordered report of the video's visual content.
    """)

    st.header("Future Options & Features (Placeholder)")
    st.markdown("""
    * **True Audio Transcription:** Integration with a dedicated ASR service for spoken words.
    * **Smart Summarization:** Advanced AI-driven summarization and keyword extraction from content.
    * **Content Translation:** Ability to translate extracted content into various languages.
    * **Speaker Identification:** Diarization to identify different speakers in the video (for audio-based content).
    * **Timestamped Events:** Generate summaries or transcripts with precise timestamps.
    * **More Input Formats:** Support for video URLs (e.g., YouTube links) in addition to file uploads.
    """)

st.title("🎥 Professional Video Visual Content Extractor & Transcriber")
st.write("Upload your video to get a comprehensive, detailed textual 'transcription' of its visual content. "
         "This app leverages the advanced visual analysis capabilities of the Google Gemini API to extract "
         "all observable information and present it in a clean, copyable, and downloadable format.")

uploaded_file = st.file_uploader(
    "Upload a video file (MP4, MOV, MKV, AVI, WEBM, etc.)",
    type=["mp4", "mov", "mkv", "avi", "webm"],
    help="Gemini API for video processing typically supports videos up to ~2 minutes. Larger files may fail or take longer."
)

transcript_display_area = st.empty()

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        video_path = tmp_file.name

    st.video(video_path)

    if st.button("Generate Content Summary", type="primary", use_container_width=True):
        with st.spinner("Analyzing video visual content with Gemini API... This may take a moment based on video length and complexity."):
            extracted_text = extract_text_with_gemini(video_path)

        with transcript_display_area.container():
            st.markdown("---")
            st.subheader("Extracted Visual Content (Visual 'Transcription')")
            st.markdown(extracted_text)

            st.text_area(
                "Copyable Visual Content Summary",
                value=extracted_text,
                height=300,
                key="copy_summary_area",
                help="You can easily copy the entire summary from this text box using the built-in copy button."
            )

            doc = Document()
            doc.add_heading('Video Visual Content Summary', level=1)
            # Split by double newline to preserve paragraphs from Gemini's output
            for paragraph_text in extracted_text.split('\n\n'):
                # Add text to the document, ensuring empty paragraphs are skipped
                if paragraph_text.strip():
                    doc.add_paragraph(paragraph_text.strip())

            bio = io.BytesIO()
            doc.save(bio)
            bio.seek(0)

            st.download_button(
                label="Download as Word (DOCX)",
                data=bio.getvalue(),
                file_name="video_visual_content_summary.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                help="Download the extracted visual content summary as a Microsoft Word document for offline use."
            )
    
    if os.path.exists(video_path):
        os.unlink(video_path)

