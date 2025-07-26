import os
from dotenv import load_dotenv
import logging

# Langchain imports
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain.chains import create_sql_query_chain # For query generation
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool # For query execution
from operator import itemgetter # For chaining operations
from langchain_core.prompts import PromptTemplate # For creating custom prompts
from langchain_core.runnables import RunnablePassthrough, RunnableParallel # For flexible chaining
from langchain_core.messages import HumanMessage, AIMessage # For chat history messages

# Import DatabaseManager from our database_utils module
from database_utils import DatabaseManager

# Setup logging for this module
try:
    from logger_config import setup_logging
    logger = setup_logging(log_level="INFO")
except ImportError:
    # Fallback logger if logger_config.py is not yet available or configured
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.warning("logger_config.py not found or not configured. Using basic logging.")


class NL2SQLChainManager:
    """
    Manages the Natural Language to SQL conversion, SQL execution,
    and natural language answer generation using Langchain components,
    with support for conversational history.
    """
    def __init__(self, db: SQLDatabase):
        """
        Initializes the NL2SQLChainManager with a Langchain SQLDatabase object
        and sets up the SQL query generation, execution, and answer generation components.

        Args:
            db (SQLDatabase): An initialized Langchain SQLDatabase object.
        """
        load_dotenv() # Ensure .env is loaded for API key

        # Retrieve OpenAI API key from environment variables
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logger.error("OPENAI_API_KEY environment variable not set.")
            raise ValueError("OPENAI_API_KEY is required for the LLM.")

        # Initialize the Large Language Model (LLM)
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)
        logger.info(f"LLM initialized: {self.llm.model_name}")

        # 1. Create the SQL Query Generation Chain
        # This chain takes a natural language question and generates a SQL query.
        # It can also take 'chat_history' as an input.
        self.sql_query_chain = create_sql_query_chain(self.llm, db)
        logger.info("Langchain SQL Query Chain created.")

        # 2. Create the SQL Database Tool for Execution
        # This tool takes a SQL query string and executes it against the database.
        self.sql_executor_tool = QuerySQLDataBaseTool(db=db)
        logger.info("Langchain SQL Database Tool for execution created.")

        # 3. Create a prompt for generating natural language answers from SQL results
        answer_prompt_template = """Given the user's question and the SQL query result,
        provide a concise and natural language answer.
        If the result is empty, state that no information was found.

        Question: {question}
        SQL Result: {sql_result}

        Natural Language Answer:"""
        self.answer_prompt = PromptTemplate.from_template(answer_prompt_template)
        logger.info("Natural language answer generation prompt created.")

        # 4. Create the Natural Language Answer Generation Chain
        self.answer_generation_chain = self.answer_prompt | self.llm.bind(stop=["\nNatural Language Answer:"])
        logger.info("Natural language answer generation chain created.")

        # Combine all parts into a single runnable sequence for history management
        # The chain flow:
        # Input to full_chain: {"question": str, "chat_history": List[BaseMessage]}
        self.full_chain = (
            RunnableParallel(
                # Pass through the original question and chat_history for the SQL generation step
                question=itemgetter("question"),
                chat_history=itemgetter("chat_history"),
                # Generate SQL query using the question and chat_history
                # create_sql_query_chain will automatically use 'question' and 'chat_history' if present
                sql_query=self.sql_query_chain
            )
            | RunnableParallel(
                # Pass original question through to the final answer generation
                question=itemgetter("question"),
                # Execute SQL and get result
                sql_result=itemgetter("sql_query") | self.sql_executor_tool
            )
            | self.answer_generation_chain # Generate natural language answer
        )
        logger.info("Full NL2SQL chain (generation + execution + answer generation) assembled with history support.")

    def process_query(self, natural_language_query: str, chat_history: list) -> str:
        """
        Processes a natural language query by generating SQL, executing it,
        and then generating a natural language answer, considering chat history.

        Args:
            natural_language_query (str): The user's current question in natural language.
            chat_history (list): A list of Langchain message objects (e.g., HumanMessage, AIMessage)
                                 representing the conversation history.

        Returns:
            str: The natural language answer to the query.
        """
        logger.info(f"Processing natural language query: '{natural_language_query}' with history.")
        try:
            # Invoke the full chain with the current natural language query and the chat history
            result = self.full_chain.invoke({
                "question": natural_language_query,
                "chat_history": chat_history
            })
            logger.info(f"Final natural language response: {result.content}")
            return result.content
        except Exception as e:
            logger.error(f"Error during query processing: {e}")
            return f"An error occurred while processing your query: {e}"