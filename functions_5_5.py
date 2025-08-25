# functions.py

import os
import asyncio
import aiofiles, aiohttp
import glob
from urllib.parse import urlparse
import pandas as pd
import logging
import json
import tempfile
from pathlib import Path
import shutil

# LangChain and AI Model imports
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from openai import AsyncOpenAI
from constants import S3_BUCKET_NAME,AWS_REGION,FAISS_INDEX_PATH
# AWS import
import aioboto3

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

# API Keys and Constants
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# --- Initialize AI Components ---
# A check to ensure keys are loaded
if not all([OPENAI_API_KEY, COHERE_API_KEY, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET_NAME]):
    raise ValueError("One or more required environment variables are not set.")

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small")
compressor = CohereRerank(cohere_api_key=COHERE_API_KEY, model="rerank-multilingual-v3.0", top_n=3)

# Load FAISS index
try:
    index = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = index.as_retriever(search_kwargs={"k": 15})
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=retriever
    )
except Exception as e:
    logger.error(f"Failed to load FAISS index from {FAISS_INDEX_PATH}: {e}")
    # You might want to handle this more gracefully in a real app
    # For now, we'll let it raise an error if the index is crucial.
    raise


# ==============================================================================
# CORE LOGIC FOR EXCEL AUTO COMPLETE
# ==============================================================================

async def _download_and_read_sources(sources: list, temp_dir: str):
    """
    Downloads source files from S3 and reads their content.
    """
    session = aioboto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )

    async def _download_file(s3_client, source):
        file_path = Path(temp_dir) / Path(source).name
        await s3_client.download_file(
            Bucket=S3_BUCKET_NAME,
            Key=source,
            Filename=str(file_path)
        )

    try:
        async with session.client('s3') as s3:
            tasks = [_download_file(s3, source) for source in sources]
            await asyncio.gather(*tasks)

        txt_files = glob.glob(f"{temp_dir}/*.txt")
        if not txt_files:
            logger.warning("No .txt files found after download.")
            return "", []

        # Combine content from all downloaded files
        full_content = ""
        for doc_path in txt_files:
            async with aiofiles.open(doc_path, "r", encoding="utf-8") as f:
                full_content += await f.read() + "\n\n"

        # Generate source URLs based on file names
        numbers = [os.path.splitext(os.path.basename(f))[0] for f in txt_files]
        source_urls = [
            f"http://www.law.go.kr/DRF/lawService.do?OC=doaz&ID={item}&target=prec&type=HTML"
            for item in numbers
        ]

        return full_content, source_urls
    except Exception as e:
        logger.error(f"Error downloading/reading sources: {e}")
        raise


async def reranked_results(query: str):
    """
    Gets reranked legal documents based on a query.
    """
    try:
        reranked_docs = await compression_retriever.ainvoke(query)
        reranked_sources = [item.metadata['source'] for item in reranked_docs]

        if not reranked_sources:
            return "I couldn't find any relevant legal documents to answer your question.", []

        # Create a temporary directory for S3 downloads
        temp_dir = Path(tempfile.mkdtemp(prefix="s3_sources_"))
        try:
            content, sources = await _download_and_read_sources(
                sources=reranked_sources,
                temp_dir=str(temp_dir)
            )
            return content, sources
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    except Exception as e:
        logger.error(f"Error in reranking: {e}")
        raise


def get_excel_data(filepath: str):
    """
    Reads and processes the uploaded Excel file.
    """
    try:
        main_df = pd.read_excel(filepath, header=1)
        # Standardize column names
        main_df.columns.values[-1] = "수행업체(관련 협력사)"
        main_df.columns.values[0] = "관리번호"

        # Select relevant columns for processing
        df_to_process = main_df.iloc[:, 1:9].copy()
        df_to_process.columns.values[5] = "시기 접수"
        df_to_process.columns.values[6] = "시기 완료"
        df_to_process.fillna("", inplace=True)

        list_of_dicts = df_to_process.to_dict(orient="records")
        return main_df, list_of_dicts
    except Exception as e:
        logger.error(f"Error reading Excel file: {e}")
        raise


async def get_ai_response(data_row: dict):
    """
    Gets AI completion for a single row of data.
    """
    try:
        # Rerank documents based on the row data
        law_data, law_source = await reranked_results(query=str(data_row))

        def excel_auto_complete_system_prompt(data, law):
            return f"""
            You are a specialized AI assistant for legal and regulatory review of construction permits. Your primary task is to analyze permit data against applicable laws and generate accurate, concise legal reviews in Korean.

            ## OBJECTIVE
            Review and populate the '관련법 검토(AI)' field for construction permit entries by cross-referencing permit requirements with applicable legal frameworks.

            ## DATA STRUCTURE
            Each permit entry contains:
            - **인허가, 심의, 평가명**: Permit/review/evaluation name
            - **정의**: Project/structure definition  
            - **관련법**: Applicable laws and provisions
            - **대상**: Target structure/project type
            - **시기 접수**: Application submission timing
            - **시기 완료**: Expected completion timing  
            - **인허가청**: Responsible regulatory authority
            - **관련법 검토(AI)**: [TO BE POPULATED] - Your legal review output

            ## JUDGMENT CRITERIA
            For each review, include appropriate status symbol:
            ✅ **Complete Match**: All requirements align perfectly with applicable laws
            ⚠️ **Partial Match**: Core framework correct, minor gaps or clarifications needed
            ❌ **Mismatch**: Significant discrepancies between stated and actual legal requirements
            🔍 **Requires Review**: Insufficient data or ambiguous requirements needing manual verification

          Your task:
          1. Carefully read and understand each entry in the list.
          2. Based on the '관련법', '대상', '시기 접수', '시기 완료', and '인허가청' fields, generate a relevant and concise legal review and populate the '관련법 검토(AI)' field.
          3. If you find '관련법', '대상', '시기 접수', '시기 완료', and '인허가청' fields are empty, leave the '관련법 검토(AI)' field empty.  
          4. Return me only 관련법 검토(AI) in JSON format.

          Here is the data:
          {data}

          For your more assistance here is law according to the data we fetch:
          {law} 
          """

        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        response = await client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": excel_auto_complete_system_prompt(data=data_row,law=law_data)},
                {"role": "user", "content": "Please review my excel file data and fill the '관련법 검토(AI)' column."}
            ],
            response_format={"type": "json_object"},
        )
        parsed_json = json.loads(response.choices[0].message.content)
        return parsed_json
    except Exception as e:
        logger.error(f"Error getting AI response for row {data_row}: {e}")
        # Return an error message in the expected format
        return {"관련법 검토(AI)": f"Error: {e}"}


async def auto_complete(filepath: str, progress_callback):
    """
    Main orchestrator function for the Streamlit app.
    Takes a filepath and a callback to update the UI.
    """
    temp_dir = None
    try:
        temp_dir = tempfile.mkdtemp(prefix="excel_output_")

        progress_callback("Reading Excel file...")
        main_df, data_rows = get_excel_data(filepath=filepath)

        progress_callback(f"Found {len(data_rows)} rows. Fetching legal data and generating AI reviews...")

        # Create and gather tasks for all rows concurrently
        tasks = [get_ai_response(row) for row in data_rows]
        json_results = await asyncio.gather(*tasks)

        progress_callback("Processing complete. Creating final Excel file...")

        # Add the new AI-generated column to the original DataFrame
        ai_review_series = pd.Series([d.get('관련법 검토(AI)', 'N/A') for d in json_results])
        main_df['관련법 검토(AI)'] = ai_review_series

        # Save the completed DataFrame to a new Excel file in the temp directory
        output_excel_path = Path(temp_dir) / "completed_review.xlsx"
        main_df.to_excel(output_excel_path, index=False)

        return {"df": main_df, "excel_file_path": str(output_excel_path), "temp_dir": temp_dir}

    except Exception as e:
        logger.error(f"An error occurred during the auto-complete process: {e}")
        # Clean up if an error occurs
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise e


def cleanup_temp_directory_excel(temp_dir: str):
    """Utility to clean up temporary directories."""
    if temp_dir and os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up temp directory {temp_dir}: {e}")
