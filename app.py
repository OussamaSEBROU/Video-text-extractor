# app.py
import streamlit as st
import os
import tempfile
from pathlib import Path
import io
from docx import Document
import google.generativeai as genai
from moviepy.editor import VideoFileClip
import time # For time.sleep during Gemini file processing

# --- Configuration ---
# Ensure the API key is loaded from environment variables for security.
# On Render.com, you would set this as an environment variable (e.g., GEMINI_API_KEY).
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("Gemini API Key not found. Please set the 'GEMINI_API_KEY' environment variable. "
             "If deploying on Render.com, add it under 'Environment Variables'.")
    st.stop() # Stop the app if the API key is missing

# Configure the Google Generative AI client with the API key
genai.configure(api_key=GEMINI_API_KEY)

# Set Streamlit page configuration for a wide layout and professional appearance
st.set_page_config(
    page_title="Video Content Extractor",
    layout="wide",
    initial_sidebar_state="expanded",
    # Favicon can be added if a local image is available, or an emoji
    # page_icon="🎥"
)

# --- Helper Functions ---

def get_video_duration(video_path):
    """
    Gets the duration of a video file in seconds using moviepy.
    This function requires `ffmpeg` to be installed and accessible in the environment
    where the Streamlit app is running.
    """
    try:
        # Load the video clip to get its duration
        clip = VideoFileClip(video_path)
        duration = clip.duration
        clip.close() # Important: Close the clip to release file resources
        return duration
    except Exception as e:
        st.error(f"Error getting video duration: {e}. "
                 "This often means `ffmpeg` is not installed or not in your system's PATH. "
                 "Please ensure `ffmpeg` is installed (e.g., `sudo apt-get install ffmpeg` on Linux, "
                 "or via Homebrew on macOS, or download binaries for Windows and add to PATH).")
        return None # Return None to indicate an error

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

    uploaded_file_name = None # To keep track of the file name on Gemini's server
    try:
        # Step 1: Upload the video file to Gemini's backend for processing
        with st.spinner("Uploading video to Gemini API..."):
            video_file = genai.upload_file(video_file_path)
            uploaded_file_name = video_file.name # Store the unique file name for future reference and deletion

        # Step 2: Wait for the file to be processed by Gemini. This is crucial as
        # Gemini needs time to ingest and prepare the video for analysis.
        processing_bar = st.progress(0, text="Processing video with Gemini API...")
        progress_percentage = 0
        while video_file.state.name == "PROCESSING":
            time.sleep(2) # Wait for 2 seconds before checking the status again
            video_file = genai.get_file(uploaded_file_name) # Refresh the file status
            # Update progress bar (this is an estimation as actual progress is not exposed)
            progress_percentage = min(progress_percentage + 5, 99)
            processing_bar.progress(progress_percentage)

        # Handle cases where video processing fails on Gemini's side
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
        model = genai.GenerativeModel('gemini-pro-vision')
        response = model.generate_content([prompt, video_file], stream=False)
        
        # Step 5: Check if the response contains valid text content
        if response and hasattr(response, 'text'):
            return response.text
        else:
            st.error("Gemini API response did not contain expected text content.")
            return "Error: Gemini API did not return valid text."

    except Exception as e:
        # Catch any exceptions during the API interaction
        st.error(f"An error occurred during Gemini API interaction: {e}")
        return "Error: Could not extract information using Gemini API."
    finally:
        # Step 6: Clean up the uploaded file from Gemini's backend to manage storage and quotas
        if uploaded_file_name:
            try:
                genai.delete_file(uploaded_file_name)
                st.success("Temporary file successfully cleaned up from Gemini API.")
            except Exception as e:
                st.warning(f"Failed to delete uploaded file from Gemini API: {e}. "
                           "Please check your Gemini API usage or manually clear if possible.")

# --- Streamlit UI Layout ---

# Sidebar content for "How to Use", "About Us", and placeholders
with st.sidebar:
    st.header("How to Use")
    st.info("""
    1.  **Upload your video file** using the "Upload a video file" button on the main page.
        (Maximum duration allowed is 15 minutes).
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

# Main Page content: Title, Introduction, File Uploader, and Output Area
st.title("🎥 Professional Video Content Extractor & Summarizer")
st.write("Upload your video to get a comprehensive textual summary of its visual content. "
         "This app leverages the advanced capabilities of the Google Gemini API to analyze your video "
         "and present insights in a clean, copyable, and downloadable format.")

# File uploader widget for video files
uploaded_file = st.file_uploader(
    "Upload a video file (MP4, MOV, MKV, AVI, WEBM, etc.)",
    type=["mp4", "mov", "mkv", "avi", "webm"], # Accepted video file types
    help="Maximum video duration allowed is 15 minutes. Please ensure sufficient internet bandwidth for upload."
)

# Placeholder for dynamically displaying the extracted content summary
transcript_display_area = st.empty()

if uploaded_file is not None:
    # Save the uploaded video to a temporary file on the server's disk
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        video_path = tmp_file.name # Get the path to the temporary file

    st.video(video_path) # Display the uploaded video in the Streamlit app

    # Check video duration before proceeding with processing
    video_duration = get_video_duration(video_path)
    if video_duration is None:
        # Error message is already displayed by get_video_duration function
        os.unlink(video_path) # Clean up the temporary file immediately
        st.stop() # Stop execution if duration cannot be determined

    if video_duration > 15 * 60: # Check against the 15-minute limit (in seconds)
        st.warning(f"Video duration ({video_duration:.2f} seconds) exceeds the 15-minute limit (900 seconds). "
                   "Please upload a shorter video.")
        os.unlink(video_path) # Clean up the temporary file
    else:
        # Button to trigger the content extraction process
        if st.button("Generate Content Summary", type="primary", use_container_width=True):
            # Show a spinner while the process runs, as it can take time
            with st.spinner("Analyzing video content with Gemini API... This may take a moment based on video length and complexity."):
                extracted_text = extract_text_with_gemini(video_path)

            # Display the extracted text in a clean, professional format
            with transcript_display_area.container():
                st.markdown("---") # Add a horizontal separator for better UI
                st.subheader("Extracted Content Summary")
                st.markdown(extracted_text) # Use st.markdown to preserve paragraph formatting

                # A copyable text area for easy interaction
                st.text_area(
                    "Copyable Summary",
                    value=extracted_text,
                    height=300,
                    key="copy_summary_area", # Unique key for the widget
                    help="You can easily copy the entire summary from this text box using the built-in copy button."
                )

                # --- Download as Word (DOCX) Option ---
                doc = Document()
                doc.add_heading('Video Content Summary', level=1)
                # Add text to the Word document, preserving paragraphs by splitting on double newlines
                for paragraph_text in extracted_text.split('\n\n'):
                    doc.add_paragraph(paragraph_text)

                # Save the document to an in-memory byte stream
                bio = io.BytesIO()
                doc.save(bio)
                bio.seek(0) # Rewind the buffer to the beginning before downloading

                # Streamlit download button for the DOCX file
                st.download_button(
                    label="Download as Word (DOCX)",
                    data=bio.getvalue(),
                    file_name="video_content_summary.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    help="Download the extracted content summary as a Microsoft Word document for offline use."
                )
    
    # Ensure the temporary video file is deleted from the local disk, regardless of success or failure
    if os.path.exists(video_path):
        os.unlink(video_path)
        # st.info(f"Temporary video file cleaned up from local storage: {video_path}") # Optional: for debugging

