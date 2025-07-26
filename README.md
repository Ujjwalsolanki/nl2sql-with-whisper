# NL2SQL Chatbot with Faster-Whisper Integration

## Project Overview

This project implements a Natural Language to SQL (NL2SQL) chatbot that allows users to query a MySQL database using plain English, either by typing or speaking. The chatbot leverages Large Language Models (LLMs) via Langchain to translate natural language questions into SQL queries, execute them against the database, and then provide the results back to the user in a conversational, natural language format. It integrates the highly optimized **Faster-Whisper** model for efficient and accurate speech-to-text transcription, enabling voice-based queries. The chatbot also maintains a limited chat history to understand contextual follow-up questions.

## Features

  * **Natural Language to SQL Conversion**: Translate user questions into executable SQL queries.
  * **Speech-to-Text with Faster-Whisper**: Convert spoken queries into text using the fast and accurate Faster-Whisper model.
  * **SQL Query Execution**: Execute the generated SQL queries against a MySQL database.
  * **Natural Language Response Generation**: Convert SQL query results into human-readable answers.
  * **Conversational History**: Maintain context for follow-up questions by remembering the last few turns of the conversation.
  * **Modular Design**: Organized into distinct Python modules for better maintainability and separation of concerns.
  * **Environment Variable Management**: Securely handle sensitive credentials (database, API keys) using `.env` files.
  * **Streamlit UI**: A simple, interactive web interface for the chatbot with both text and audio input options.

## Technologies Used

  * **Python 3.x**
  * **Streamlit**: For building the interactive web UI.
  * **Langchain**: Framework for building LLM-powered applications.
      * `langchain-community`
      * `langchain-core`
      * `langchain-openai`
  * **SQLAlchemy**: Python SQL Toolkit and Object Relational Mapper (ORM) for database interactions.
  * **PyMySQL**: MySQL database connector for Python (used by SQLAlchemy).
  * **python-dotenv**: For loading environment variables.
  * **OpenAI API**: For the Large Language Model (LLM) capabilities (GPT-3.5 Turbo).
  * **Faster-Whisper**: For optimized speech-to-text transcription.
  * **Soundfile**: For reading and writing audio files.
  * **NumPy**: For numerical operations on audio data.
  * **SciPy**: For audio resampling (`scipy.signal.resample_poly`).

## Project Structure

```
nl2sql_project/
├── .env                  # Environment variables (DB credentials, OpenAI API Key)
├── requirements.txt      # Python dependencies
├── app.py                # Streamlit application for the UI
├── database_utils.py     # Module for database connection and Langchain SQLDatabase setup
├── llm_chain.py          # Module for defining the Langchain NL2SQL chain (query generation, execution, answer generation)
├── logger_config.py      # Module for configuring application logging
├── speech_to_text.py     # NEW: Module for Faster-Whisper integration (audio transcription)
├── README.md             # Project description and setup instructions
├── logs/                 # Directory for application logs (created automatically)
│   └── nl2sql_app_YYYYMMDD_HHMMSS.log # Timestamped log file
└── models/               # Directory for downloaded Faster-Whisper models
    └── tiny/             # Example: 'tiny' model files (or 'large-v2', 'Systran--faster-whisper-large-v2')
        ├── config.json
        ├── model.bin
        └── ...
└── mysql_db/             # Optional: Directory for MySQL database dump files
    └── your_database.sql # Example SQL dump file

```

## Setup and Installation

1.  **Clone the repository (if applicable):**

    ```bash
    git clone <your-repo-url>
    cd nl2sql_project
    ```

2.  **Create a Python Virtual Environment:**
    It's highly recommended to use a virtual environment to manage dependencies.

    ```bash
    python -m venv venv
    ```

3.  **Activate the Virtual Environment:**

      * **On macOS/Linux:**

        ```bash
        source venv/bin/activate
        ```

      * **On Windows:**

        ```bash
        venv\Scripts\activate
        ```

4.  **Install Dependencies:**
    Install all required Python packages using `pip`. Ensure your `requirements.txt` includes `faster-whisper`, `soundfile`, `numpy`, and `scipy`.

    ```bash
    pip install -r requirements.txt
    ```

5.  **Install `libsndfile` (System Dependency for `soundfile`):**
    `soundfile` relies on the `libsndfile` system library. Install it using your system's package manager:

      * **On Ubuntu/Debian:** `sudo apt-get update && sudo apt-get install libsndfile1`
      * **On Fedora/RHEL:** `sudo dnf install libsndfile`
      * **On macOS (with Homebrew):** `brew install libsndfile`
      * **On Windows:** Installation via `pip install soundfile` often includes necessary binaries. If issues persist, you might need to manually download `libsndfile` DLLs and add them to your system's PATH.

6.  **Download Faster-Whisper Model:**
    Download the specific Faster-Whisper model you intend to use (e.g., `tiny`, `base`, `small`, `medium`, `large-v2`, or `Systran/faster-whisper-large-v2`). Place the downloaded model files (the entire folder containing `config.json`, `model.bin`, etc.) into the `./models/` directory within your project.
    For example, if you're using `tiny`, the path should be `./models/tiny/`. If you're using `Systran--faster-whisper-large-v2`, it should be `./models/Systran--faster-whisper-large-v2/`.

7.  **Configure Environment Variables (`.env` file):**
    Create a file named `.env` in the root of your `nl2sql_project` directory and add your MySQL database credentials and OpenAI API Key. Replace the placeholder values with your actual information.

    ```dotenv
    # .env
    DB_USER="root" # Or your MySQL username
    DB_PASSWORD="your_mysql_root_password" # Your actual MySQL password
    DB_HOST="localhost"
    DB_PORT="3306"
    DB_NAME="classicmodels" # Your specific database name

    OPENAI_API_KEY="sk-your_openai_api_key_here" # Your actual OpenAI API Key

    # Faster-Whisper Configuration (optional, if you want to override defaults in speech_to_text.py)
    # WHISPER_MODEL_TYPE="tiny" # e.g., "tiny", "base", "small", "medium", "large-v2"
    # WHISPER_RUN_TYPE="cpu" # or "gpu"
    # WHISPER_GPU_INDICES="0" # Comma-separated for multiple GPUs, e.g., "0,1"
    # WHISPER_NUM_WORKERS="10" # Number of workers for CPU
    # WHISPER_CPU_THREADS="4" # Number of CPU threads for CPU
    ```

8.  **Ensure MySQL Database is Running:**
    Make sure your MySQL server is running and the specified `DB_NAME` database exists and is accessible with the provided credentials.

9.  **Load Database from `mysql_db` Folder (if applicable):**
    If your database schema and data are provided as a SQL dump file (e.g., `your_database.sql`) in a `mysql_db` folder, you can load it into your MySQL instance using the `mysql` command-line client.

      * **Navigate to your project's root directory.**

      * **Execute the SQL dump:**

        ```bash
        mysql -u your_mysql_username -p your_database_name < mysql_db/your_database.sql
        ```

        (Replace `your_mysql_username` and `your_database_name` with your actual MySQL username and the database name you configured in `.env`. You will be prompted for your MySQL password.)

## Usage

1.  **Run the Streamlit Application:**
    With your virtual environment activated, run the `app.py` file:

    ```bash
    streamlit run app.py
    ```

2.  **Interact with the Chatbot:**
    Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`). You can type your natural language questions into the chat input or use the audio recording widget to speak your queries.

## Interview Focus / Key Learnings

This project demonstrates several key software engineering and AI development concepts that are frequently discussed in technical interviews:

1.  **Modular Design and Separation of Concerns**:

      * **`database_utils.py`**: Handles all database connection logic.
      * **`llm_chain.py`**: Contains the core LLM and Langchain logic.
      * **`logger_config.py`**: Centralizes logging configuration.
      * **`speech_to_text.py`**: Encapsulates the speech-to-text functionality, making it easy to swap transcription models.
      * **`app.py`**: Focuses solely on the Streamlit UI and orchestrating calls to the backend modules.

2.  **Dependency Management**:

      * `requirements.txt` ensures reproducibility.
      * `venv` isolates project dependencies.

3.  **Configuration Management and Security**:

      * Using a `.env` file with `python-dotenv` for sensitive information (API keys, database credentials).

4.  **Langchain Fundamentals**:

      * `SQLDatabase`, `create_sql_query_chain`, `QuerySQLDataBaseTool`.
      * Runnable Interface (`|`, `RunnableParallel`, `RunnablePassthrough`) for building complex LLM applications.
      * Prompt Engineering.

5.  **Speech-to-Text Integration**:

      * Demonstrates integrating a specialized, high-performance STT model (`faster-whisper`) into a Streamlit application.
      * Handling audio input from a web interface (`st.audio_input`) and preparing it for a local STT model (e.g., resampling, format conversion using `soundfile`).
      * Managing model loading and resource usage for local inference.

6.  **State Management in Streamlit**:

      * The use of `st.session_state` is critical for maintaining `DatabaseManager`, `NL2SQLChainManager`, `SpeechToTextManager`, and chat `messages` across Streamlit's reruns, preventing expensive re-initializations and preserving conversational context.
      * Techniques like dynamic `key` for `st.audio_input` to enable programmatic clearing.

7.  **Error Handling and Robustness**:

      * `try-except` blocks are implemented in all modules to gracefully handle potential issues.
      * Logging provides visibility into the application's runtime behavior.

8.  **Conversational AI Concepts**:

      * Chat History Management for contextual understanding.
      * Input prioritization (text vs. audio) for a flexible user experience.

9.  **Scalability and Extensibility (Future Considerations)**:

      * The modular design allows for easy swapping of components (e.g., LLM providers, database types, STT models).
      * Foundation for advanced features like few-shot examples, input validation, or more sophisticated error recovery.