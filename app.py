import streamlit as st
import os
import tempfile
from pathlib import Path
import io
from docx import Document
import google.generativeai as genai
import time # For time.sleep during AI file processing

# --- Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("AI service key not found. Please set the 'GEMINI_API_KEY' environment variable.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

st.set_page_config(
    page_title="TahiriExtractor - Video Ultra Transcription",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Initialize Session State for Chat ---
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = ""
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'page_selection' not in st.session_state:
    st.session_state.page_selection = "Video Extraction" # Default page

# --- Helper Function for AI Video Processing ---

def extract_text_with_ai(video_file_path):
    """
    Extracts comprehensive visual content and on-screen text from a video
    using an advanced AI model. This function aims for a "transcription-like"
    output based purely on visual observation.

    Important Note on Transcription:
    This model excels at understanding visual content and reading on-screen text.
    It **does NOT perform direct audio-to-text transcription**. The output will be
    a detailed descriptive summary of the video's visual elements and any text
    appearing in frames, structured to resemble a continuous narrative or report.
    For true spoken word transcription, a dedicated ASR service is required.
    """
    st.info("Initiating video analysis with our AI for visual content extraction. "
            "Please note: This process focuses on what is *visually observable* in the video, "
            "including any on-screen text, and infers narrative from visual cues. "
            "It does NOT perform audio-to-text transcription.")

    uploaded_file_name = None
    try:
        with st.spinner("Uploading video for AI analysis..."):
            video_file = genai.upload_file(video_file_path)
            uploaded_file_name = video_file.name

        processing_bar = st.progress(0, text="Processing video with AI... Please wait, this may take a moment.")
        progress_percentage = 0
        while video_file.state.name == "PROCESSING":
            time.sleep(2)
            video_file = genai.get_file(uploaded_file_name)
            progress_percentage = min(progress_percentage + 5, 99)
            processing_bar.progress(progress_percentage)

        if video_file.state.name == "FAILED":
            st.error("Failed to process video with our AI. Please try again or with a different video.")
            return "Error: Video processing failed."

        # --- Refined Prompt for Direct Content Extraction (UNMODIFIED AS PER INSTRUCTION) ---
        prompt = (
            "You are a highly precise content extraction AI. Your task is to extract the text from the video without change anything thing in the content"
" the language of extracted text should be same of the video .. be carefuly on that"

            "Present the extracted content in clean, well-formatted paragraphs, ordering events chronologically as they appear in the video. "
           
        )
        
        # Using gemini-2.0-flash model (UNMODIFIED AS PER INSTRUCTION)
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content([prompt, video_file], stream=False)
        
        if response and hasattr(response, 'text'):
            return response.text
        else:
            st.error("Our AI service did not return expected text content.")
            return "Error: AI analysis did not return valid text."

    except Exception as e:
        st.error(f"An error occurred during AI analysis: {e}")
        return "Error: Could not extract information using our AI service."
    finally:
        if uploaded_file_name:
            try:
                genai.delete_file(uploaded_file_name)
                st.success("Temporary file successfully cleaned up from our AI system.")
            except Exception as e:
                st.warning(f"Failed to delete uploaded file from our AI system: {e}. "
                           "Please check your usage or manually clear if possible.")

# --- Helper Function for Chat with Extracted Text ---
def chat_with_extracted_text(user_query):
    # Specific response for identity questions
    identity_keywords = ["who are you", "who developed you", "what technology", "who made you", "your developer", "your creator"]
    if any(keyword in user_query.lower() for keyword in identity_keywords):
        return "TahiriExtractor AI-chat assistance, developed by TahiriExtractor team, use LLMs technology."

    if not st.session_state.extracted_text:
        return "Please extract video content first to enable chat functionality."

    # Construct chat history for the model, focusing on the extracted text as context
    chat_model = genai.GenerativeModel('gemini-2.0-flash')
    
    # Initialize chat history with a system prompt and the extracted text as initial context
    # This ensures the model always has the extracted text as the primary context
    messages = [
        {"role": "user", "parts": [
            "You are TahiriExtractor AI-chat assistance. Your sole purpose is to answer questions strictly based on the provided video content summary. Provide professional, deep, and concise answers, focusing solely on analysis of the text. Do not answer questions outside of this context. If a user asks about your identity, development, or technology, respond with: 'TahiriExtractor AI-chat assistance, developed by TahiriExtractor team, use LLMs technology.' Do not deviate from this specific response for such queries."
            f"Here is the video content summary: {st.session_state.extracted_text}"
        ]}
    ]

    # Add the user's current query
    messages.append({"role": "user", "parts": [user_query]})

    try:
        response = chat_model.generate_content(messages)
        if response and hasattr(response, 'text'):
            return response.text
        else:
            return "Could not generate a response based on the provided text."
    except Exception as e:
        return f"An error occurred during chat: {e}"

# --- Custom CSS for Theming and Chat Styling ---
st.markdown("""
<style>
    /* General body styling for theme compatibility */
    body {
        color: var(--text-color); /* Streamlit's default text color */
        background-color: var(--background-color); /* Streamlit's default background color */
        font-family: 'Inter', sans-serif; /* Apply Inter font globally */
    }

    /* Main Title Container */
    .main-title-container {
        padding: 30px 0;
        background: var(--background-color-secondary); /* Adapts to Streamlit theme */
        border-radius: 15px;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1); /* Subtle shadow, generally fine for both themes */
        text-align: center;
    }
    .main-title {
        color: #0A6EFD; /* A more distinct, vibrant professional blue */
        font-size: 2.8em;
        font-weight: 700;
        letter-spacing: 1px;
        margin-bottom: 10px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.05); /* Very subtle, lightens on dark */
    }
    .main-subtitle {
        color: var(--text-color-secondary); /* Adapts to Streamlit theme */
        font-size: 1.2em;
        max-width: 800px;
        margin: 0 auto;
        line-height: 1.6;
    }
    .info-box {
        background-color: var(--background-color-tertiary); /* Adapts to Streamlit theme */
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-top: 25px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    .info-box p {
        font-size: 1.05em;
        color: var(--text-color); /* Adapts to Streamlit theme */
    }
    .info-box strong {
        color: #0A6EFD; /* Consistent vibrant blue emphasis */
    }

    /* Extracted Text Output Styling (no subheader above it) */
    .extracted-text-output {
        background-color: var(--background-color-secondary); /* Adapts to Streamlit theme */
        padding: 20px;
        border-radius: 12px;
        border-left: 6px solid #0A6EFD; /* Stronger blue border */
        margin-bottom: 25px;
        overflow-wrap: break-word;
        font-family: 'Inter', sans-serif;
        line-height: 1.6;
        color: var(--text-color); /* Adapts to Streamlit theme */
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    /* Video Player Styling */
    .stVideo {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
        margin-bottom: 25px;
    }

    /* Chat Container for scrolling */
    .chat-messages-container {
        max-height: 400px; /* Fixed height for scrolling chat */
        overflow-y: auto; /* Enable vertical scrolling */
        padding-right: 15px; /* Space for scrollbar */
        padding-top: 10px;
        padding-bottom: 10px;
        margin-bottom: 15px;
        border-radius: 10px;
        background-color: var(--background-color-tertiary); /* Subtle background for chat area */
        box-shadow: inset 0 0 5px rgba(0,0,0,0.05); /* Inner shadow */
    }

    /* Chat Message Styling (ChatGPT-like) */
    .chat-message-user, .chat-message-ai {
        padding: 12px 18px;
        border-radius: 18px; /* More rounded like chat bubbles */
        margin-bottom: 8px; /* Slightly less margin for denser chat */
        max-width: 80%; /* Limits bubble width */
        font-size: 0.95em;
        line-height: 1.5;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08); /* Subtle shadow */
        word-wrap: break-word; /* Ensure long words break */
    }
    .chat-message-user {
        background-color: var(--primary-color-20); /* Light blue/green for user messages */
        color: var(--text-color);
        margin-left: auto; /* Aligns user messages to the right */
        border-bottom-right-radius: 4px; /* Sharpen one corner */
    }
    .chat-message-ai {
        background-color: var(--secondary-background-color); /* Streamlit's secondary bg for AI messages */
        color: var(--text-color);
        margin-right: auto; /* Aligns AI messages to the left */
        border-bottom-left-radius: 4px; /* Sharpen one corner */
    }
    .stTextInput label { /* Style for chat input label */
        font-weight: 600;
        color: var(--text-color);
    }
    .stTextInput > div > div > input { /* Style for chat input field */
        border-radius: 10px;
        padding: 10px;
        background-color: var(--background-color-secondary);
        color: var(--text-color);
    }
</style>

<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">

<div class="main-title-container">
    <h1 class="main-title">TahiriExtractor - Video Ultra Transcription</h1>
    <p class="main-subtitle">
        Your ultimate AI-powered tool for extracting comprehensive visual information and on-screen text from videos.
    </p>
    <div class="info-box">
        <p>Harnessing <strong>advanced Artificial Intelligence</strong> to deliver detailed, readable summaries.</p>
        <p><strong>Key Point:</strong> Supports videos up to <strong>15 minutes</strong> in duration for efficient analysis. Focuses on visual content; does not transcribe audio.</p>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Sidebar Navigation ---
st.sidebar.header("Navigation")
page_selection = st.sidebar.radio(
    "Go to",
    ("Video Extraction", "Chat with Content", "About TahiriExtractor", "Contact Us"),
    key="page_selector"
)
st.session_state.page_selection = page_selection


# --- Main Content Area ---
if st.session_state.page_selection == "Video Extraction":
    st.subheader("Extract Visual Content")
    uploaded_file = st.file_uploader(
        "Upload a video file (MP4, MOV, MKV, AVI, WEBM, etc.)",
        type=["mp4", "mov", "mkv", "avi", "webm"],
        help="Our AI system supports videos up to approximately 15 minutes in length. Larger files may fail or take longer."
    )

    transcript_display_area = st.empty()

    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            video_path = tmp_file.name

        st.video(video_path)

        if st.button("Generate Content Summary", type="primary", use_container_width=True):
            with st.spinner("Analyzing video visual content with our AI... This may take a moment based on video length and complexity."):
                extracted_text = extract_text_with_ai(video_path)
                st.session_state.extracted_text = extracted_text # Store extracted text in session state
                st.session_state.chat_history = [] # Reset chat history when new text is extracted

            with transcript_display_area.container():
                st.markdown("---")
                
                # Display the extracted text directly without an introduction/subheader
                st.markdown(
                    f"""
                    <div class="extracted-text-output">
                        {extracted_text}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Copyable text area (with built-in copy icon)
                st.text_area(
                    "Copyable Visual Content Summary",
                    value=extracted_text,
                    height=300,
                    key="copy_summary_area",
                    help="You can easily copy the entire summary from this text box using the built-in clipboard icon (top-right of this text area)."
                )

                # Create and download Word document
                doc = Document()
                doc.add_heading('Video Visual Content Summary', level=1)
                for paragraph_text in extracted_text.split('\n\n'):
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
        
        # Ensure temporary file is cleaned up after use
        if os.path.exists(video_path):
            os.unlink(video_path)

elif st.session_state.page_selection == "Chat with Content":
    st.subheader("Chat with Extracted Content")
    if not st.session_state.extracted_text:
        st.info("Please go to 'Video Extraction' to upload a video and generate content first to enable chat.")
    else:
        st.write("Ask questions about the video content that was just extracted.")

        # Chat messages container with scrolling
        st.markdown("<div id='chat-messages-scroll-container' class='chat-messages-container'>", unsafe_allow_html=True)
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f"<div class='chat-message-user'><b>You:</b> {message['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='chat-message-ai'><b>TahiriExtractor AI:</b> {message['content']}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True) # Close chat messages container

        # JavaScript to scroll to the bottom of the chat container
        st.markdown("""
        <script>
            var objDiv = document.getElementById("chat-messages-scroll-container");
            if (objDiv) {
                objDiv.scrollTop = objDiv.scrollHeight;
            }
        </script>
        """, unsafe_allow_html=True)

        user_chat_query = st.text_input("Your question about the content:", key="chat_input")

        if st.button("Ask TahiriExtractor AI", type="secondary"):
            if user_chat_query:
                st.session_state.chat_history.append({"role": "user", "content": user_chat_query})
                with st.spinner("TahiriExtractor AI is thinking..."):
                    ai_response = chat_with_extracted_text(user_chat_query)
                    st.session_state.chat_history.append({"role": "ai", "content": ai_response})
                st.rerun()
            else:
                st.warning("Please enter a question to chat.")

elif st.session_state.page_selection == "About TahiriExtractor":
    st.subheader("About TahiriExtractor")
    st.markdown("""
    **TahiriExtractor** is an innovative application leveraging **advanced Artificial Intelligence** to:
    * Extract deep textual insights from video visual content.
    * Analyze video frames for a comprehensive, structured summary.
    * Capture and transcribe any on-screen text, actions, and visual narratives.
    * **Important Note:** This application does not transcribe audio content (speech-to-text).

    Our mission is to provide an efficient and powerful tool for researchers, content creators, and professionals
    who need to quickly understand and utilize the visual narrative embedded in video content.
    """)

elif st.session_state.page_selection == "Contact Us":
    st.subheader("Contact Us")
    st.markdown("""
    Have questions, feedback, or need support? We'd love to hear from you!

    Please feel free to reach out to our team. We are committed to providing excellent support and continuously improving TahiriExtractor.

    Contact us via email:
    [TahiriExtractor.veo.net](mailto:oussama.sebrou@gmail.com?subject=Professional%20Inquiry%20regarding%20TahiriExtractor%20Application&body=Dear%20TahiriExtractor%20Team%2C%0A%0AI%20am%20writing%20to%20you%20from%20the%20TahiriExtractor%20application.%20My%20inquiry%20is%20regarding%3A%20%5Bbriefly%20state%20purpose%5D%0A%0AThank%20you%20for%20your%20time%20and%20assistance.%0A%0ASincerely%2C%0A%5BYour%20Name%5D)
    """)

