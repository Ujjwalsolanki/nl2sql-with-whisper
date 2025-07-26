# app.py

import streamlit as st
import logging
from langchain_core.messages import HumanMessage, AIMessage
import io # Needed for audio processing
import asyncio # Needed to run async functions

# Import our custom modules
from database_utils import DatabaseManager
from llm_chain import NL2SQLChainManager
from logger_config import setup_logging
# Import the ChunkedSpeechToTextManager class
from chunked_speech_to_text import ChunkedSpeechToTextManager

# Setup logging for the Streamlit app
logger = setup_logging(log_file_prefix="nl2sql_app", log_level="INFO", console_output=True)

# --- Streamlit UI Configuration ---
st.set_page_config(
    page_title="NL2SQL Chatbot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.title("🤖 NL2SQL Chatbot")
st.markdown("Ask me questions about your database in natural language, either by typing or speaking!")

# --- Initialize Session State ---
# This helps prevent re-initializating heavy objects (like LLM chains) on every rerun.
if "db_manager" not in st.session_state:
    try:
        st.session_state.db_manager = DatabaseManager()
        st.session_state.langchain_sql_db = st.session_state.db_manager.get_langchain_db()
        logger.info("DatabaseManager and Langchain SQLDatabase initialized in session state.")
    except Exception as e:
        st.error(f"Failed to connect to the database: {e}")
        logger.critical(f"Failed to initialize DatabaseManager: {e}", exc_info=True)
        st.stop() # Stop the app if DB connection fails

if "nl2sql_chain" not in st.session_state:
    try:
        st.session_state.nl2sql_chain = NL2SQLChainManager(st.session_state.langchain_sql_db)
        logger.info("NL2SQLChainManager initialized in session state.")
    except Exception as e:
        st.error(f"Failed to initialize the NL2SQL chain: {e}")
        logger.critical(f"Failed to initialize NL2SQLChainManager: {e}", exc_info=True)
        st.stop() # Stop the app if LLM chain fails

# NEW: Initialize ChunkedSpeechToTextManager
if "chunked_stt_manager" not in st.session_state:
    try:
        st.session_state.chunked_stt_manager = ChunkedSpeechToTextManager()
        logger.info("ChunkedSpeechToTextManager initialized in session state.")
    except Exception as e:
        st.error(f"Failed to initialize Speech-to-Text service: {e}")
        logger.critical(f"Failed to initialize ChunkedSpeechToTextManager: {e}", exc_info=True)
        st.stop() # Stop the app if STT fails

if "messages" not in st.session_state:
    st.session_state.messages = []
    logger.info("Chat history initialized in session state.")

# Initialize a key for the audio_input widget to allow programmatic clearing
if "audio_input_key" not in st.session_state:
    st.session_state.audio_input_key = 0

# --- Display Chat Messages ---
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# --- Clear History Button ---
# This button will appear just above the input fields.
if st.button("Clear Chat History", key="clear_chat_button"):
    st.session_state.messages = []
    logger.info("Chat history cleared.")
    st.session_state.audio_input_key += 1 # Clear audio input
    st.rerun() # Rerun the app to clear the displayed messages and inputs

# --- User Input Area ---
# Place audio_input directly above chat_input.
# st.chat_input will pull everything below it to the bottom of the screen.

# Audio Input using st.audio_input
# We use a dynamic key to allow clearing the widget after processing
audio_file_uploaded = st.audio_input(
    label="Record your question", # Label is required
    key=f"audio_input_{st.session_state.audio_input_key}" # Dynamic key
)

# Text Input - Using st.chat_input, which automatically goes to the very bottom
text_prompt = st.chat_input("Ask a question about your database...")


# --- Determine the actual prompt to use and process it ---
prompt_to_process = None
should_rerun = False # Flag to control rerunning

# Case 1: Text input is provided
if text_prompt: # st.chat_input returns a non-empty string on submission
    prompt_to_process = text_prompt
    logger.info(f"Processing text input from chat_input: '{text_prompt}'")
    # Increment audio_input_key to clear the audio widget on the next rerun
    st.session_state.audio_input_key += 1
    logger.info("Text input detected, incrementing audio_input_key to clear audio widget.")
    should_rerun = True # Mark for rerun after processing

# Case 2: Audio input is provided (and no text input was submitted in this run)
elif audio_file_uploaded is not None and audio_file_uploaded.size > 0:
    with st.spinner("Transcribing audio..."):
        try:
            # Run the async transcription function from the ChunkedSpeechToTextManager instance
            transcribed_text = asyncio.run(st.session_state.chunked_stt_manager.transcribe_audio_async_chunked(audio_file_uploaded, chunk_length_s=5))
            if transcribed_text:
                prompt_to_process = transcribed_text
                logger.info(f"Audio transcribed: '{transcribed_text}'")
            else:
                st.warning("Could not transcribe audio. Please try speaking more clearly.")
                logger.warning("Audio transcription returned empty string.")
        except Exception as e:
            st.error(f"Error during audio transcription: {e}")
            logger.error(f"Error transcribing audio: {e}", exc_info=True)
            prompt_to_process = None # Ensure prompt is not processed if transcription fails
    # Always clear audio input after processing, whether successful or not
    st.session_state.audio_input_key += 1
    logger.info("Audio processed, incrementing audio_input_key to clear audio widget.")
    should_rerun = True # Mark for rerun after processing


# --- Process the prompt if available (either from text or transcribed audio) ---
# This block will now execute only if prompt_to_process was successfully determined
if prompt_to_process:
    # Add user message to chat history
    st.session_state.messages.append(HumanMessage(content=prompt_to_process))

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Limit chat_history to the last 10 messages (5 user, 5 AI) for context
                context_history = st.session_state.messages[:-1] # Exclude current HumanMessage
                if len(context_history) > 10:
                    context_history = context_history[-10:]

                response = st.session_state.nl2sql_chain.process_query(
                    natural_language_query=prompt_to_process,
                    chat_history=context_history
                )
                st.markdown(response)
                st.session_state.messages.append(AIMessage(content=response))

                # After appending the new AI message, ensure total history length is limited
                if len(st.session_state.messages) > 10:
                    st.session_state.messages = st.session_state.messages[-10:]
                    logger.info("Chat history truncated to last 10 messages (5 pairs).")

            except Exception as e:
                error_message = f"An error occurred while processing your request. Please try again. Error: {e}"
                st.error(error_message)
                logger.error(f"Error during Streamlit query processing: {e}", exc_info=True)
                st.session_state.messages.append(AIMessage(content="Sorry, I encountered an error. Please check the logs."))
                # Ensure history is still limited even on error
                if len(st.session_state.messages) > 10:
                    st.session_state.messages = st.session_state.messages[-10:]

    # Only rerun if a prompt was successfully processed and an LLM response was generated
    # This ensures the LLM call completes before the UI updates.
    if should_rerun:
        st.rerun()
