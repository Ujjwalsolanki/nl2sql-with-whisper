from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import Engine
from langchain_community.utilities import SQLDatabase
import os
from dotenv import load_dotenv # Import load_dotenv
import logging

# Setup logging for this module
# We'll assume logger_config.py will be imported and setup_logging called elsewhere
# For standalone execution, we'll set up a basic logger here.
try:
    from logger_config import setup_logging
    logger = setup_logging(log_level="INFO")
except ImportError:
    # Fallback logger if logger_config.py is not yet available or configured
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.warning("logger_config.py not found or not configured. Using basic logging.")


class DatabaseManager:
    """
    Manages database connection and provides Langchain SQLDatabase object.
    Handles loading credentials from .env and constructing the database URI.
    """
    def __init__(self):
        """
        Initializes the DatabaseManager, loads credentials from .env,
        constructs the database URI, connects to the database,
        and creates the Langchain SQLDatabase object.
        """
        # Load environment variables from .env file
        load_dotenv()

        # Retrieve database credentials from environment variables
        db_user = os.getenv("DB_USER")
        db_password = os.getenv("DB_PASSWORD")
        db_host = os.getenv("DB_HOST")
        db_port = os.getenv("DB_PORT", "3306") # Default to 3306 if not specified
        db_name = os.getenv("DB_NAME")

        # Construct the DATABASE_URI using environment variables
        if not all([db_user, db_password, db_host, db_name]):
            logger.error("Error: Database credentials (DB_USER, DB_PASSWORD, DB_HOST, DB_NAME) must be set in your .env file.")
            raise ValueError("Missing database credentials in .env file.")

        self.database_uri = (
            f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        )

        self.engine = None
        self.langchain_db = None
        self._connect_to_db()

    def _connect_to_db(self):
        """
        Establishes the SQLAlchemy engine connection and creates the Langchain SQLDatabase object.
        This method is called automatically during object initialization.
        """
        try:
            self.engine = create_engine(self.database_uri)
            # The Langchain SQLDatabase initialization below will implicitly test the connection.
            logger.info(f"Attempting to connect to the database: {self.database_uri}")

            self.langchain_db = SQLDatabase(self.engine)
            logger.info("Langchain SQLDatabase object created and connection established.")

        except Exception as e:
            logger.error(f"Error connecting to the database or creating Langchain SQLDatabase object: {e}")
            self.engine = None # Ensure engine is None if connection fails
            self.langchain_db = None # Ensure langchain_db is None if connection fails
            raise # Re-raise the exception to indicate failure to the caller

    def get_engine(self) -> Engine:
        """
        Returns the SQLAlchemy engine object.

        Returns:
            Engine: The SQLAlchemy engine object.
        """
        if not self.engine:
            raise ConnectionError("Database engine is not initialized. Connection might have failed.")
        return self.engine

    def get_langchain_db(self) -> SQLDatabase:
        """
        Returns the Langchain SQLDatabase object.

        Returns:
            SQLDatabase: The Langchain SQLDatabase object.
        """
        if not self.langchain_db:
            raise ConnectionError("Langchain SQLDatabase object is not initialized. Connection might have failed.")
        return self.langchain_db

    def get_database_schema(self) -> dict:
        """
        Retrieves and returns the schema of the connected database.

        Returns:
            dict: A dictionary containing table names and their column information.
        """
        if not self.engine:
            raise ConnectionError("Cannot retrieve schema: Database engine is not initialized.")
        
        inspector = inspect(self.engine)
        schema = {}
        try:
            table_names = inspector.get_table_names()
            logger.info(f"Found tables: {table_names}")
            for table_name in table_names:
                columns = inspector.get_columns(table_name)
                schema[table_name] = [{"name": col['name'], "type": str(col['type'])} for col in columns]
            logger.info("Database schema retrieved.")
            return schema
        except Exception as e:
            logger.error(f"Error retrieving database schema: {e}")
            raise