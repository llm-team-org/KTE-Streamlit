import asyncio
import os
import json
from pathlib import Path
import logging
import base64
import io
import tempfile

import fitz  # PyMuPDF
import docx2txt
from PIL import Image
from llama_parse import LlamaParse
from openai import AsyncOpenAI

from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

def multi_doc_query_system_prompt(query: str, doc_data: str):
    return f"""
        You are an intelligent document analysis assistant.

        Inputs:
        - Query: ***{query}*** (The question or item to look for)
        - Documents: ***{doc_data}*** (A document contents to analyze)

        Task:
        1. Analyze document completely
        2. For the query, extract the exact factual answer from the document.
        3. If the information is not mentioned, return "Not specified".
        4. Return the output as a valid JSON string that can be used directly in Python.

        Rules:
        - No extra explanations or text—return only the JSON string.
        - Ignore all formatting symbols, icons, or meta-comments (e.g., ✅, ⚠️).

        Expected Python JSON string format:
        '{{
          "answer": "<answer from document>"
        }}'
        """
# --- Helper Functions for Text Extraction ---

async def extract_text_from_docx(file_path: str):
    """Extracts text from a DOCX file using docx2txt."""
    try:
        text = docx2txt.process(file_path)
        return text
    except Exception as e:
        logger.error(f"Error extracting text from DOCX: {e}")
        raise e


async def extract_text_from_excel_with_llama(file_path: str):
    """Extracts text from all sheets of an XLSX file using LlamaParse."""
    parser = LlamaParse(
        api_key=LLAMA_CLOUD_API_KEY,
        result_type="text",
        system_prompt="Extract all text content from every sheet in the Excel file."
    )
    try:
        documents = await parser.aload_data(file_path)
        all_text = "\n\n--- New Sheet ---\n\n".join(
            doc.text.strip() for doc in documents if doc.text and doc.text.strip()
        )
        return all_text
    except Exception as e:
        logger.error(f"Error extracting text from Excel: {e}")
        raise e


async def _clean_output(text: str):
    """Removes common introductory phrases from AI output."""
    prefixes_to_remove = [
        "Here's the extracted text from the image:", "Here is all the text extracted from the image:",
        "Here is the extracted text from the image:", "The extracted text from the image is:",
        "Extracted text:", "Here's the text:", "Here is the text:"
    ]
    try:
        cleaned_text = text.strip()
        for prefix in prefixes_to_remove:
            if cleaned_text.startswith(prefix):
                cleaned_text = cleaned_text[len(prefix):].strip()
                break
        return cleaned_text
    except Exception as e:
        logger.error(f"Error cleaning output: {e}")
        raise e


async def pdf_to_base64_images(pdf_path: str, dpi: int = 300):
    """Convert PDF pages directly to base64 data URIs."""
    doc = fitz.open(pdf_path)
    base64_images = []

    try:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=dpi, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Convert to base64
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
            base64_images.append(f"data:image/png;base64,{img_str}")
    finally:
        doc.close()

    return base64_images


async def extract_text_from_pdf_with_openai(img):
    """Extract text from a single PDF page image using OpenAI."""
    try:
        completion = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Do OCR and Extract all text and symbols from this image."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": img,
                            },
                        },
                    ],
                }
            ],
        )
        cleaned_text = await _clean_output(completion.choices[0].message.content)
        return cleaned_text
    except Exception as e:
        logger.error(f"Error in PDF page extraction: {e}")
        raise e


async def pdf_text_extractor(pdf_path: str):
    """Extract text from all pages of a PDF using OpenAI OCR."""
    try:
        all_text = ""
        batch_size = 30
        concurrent_limit = 30

        # Create a semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(concurrent_limit)

        async def extract_with_semaphore(img):
            async with semaphore:
                await asyncio.sleep(0.5)
                return await extract_text_from_pdf_with_openai(img=img)

        all_images = await pdf_to_base64_images(pdf_path=pdf_path)

        for i in range(0, len(all_images), batch_size):
            batch = all_images[i:i + batch_size]
            logger.info(f"Working on Batch {i // batch_size + 1}/{(len(all_images) + batch_size - 1) // batch_size}")

            try:
                tasks = []
                for idx, img in enumerate(batch):
                    await asyncio.sleep(0.5)
                    logger.info(
                        f"Processing image {idx + 1}/{len(batch)} in current batch (overall image {i + idx + 1})")
                    tasks.append(asyncio.create_task(extract_with_semaphore(img)))

                batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                # Handle individual failures with retry
                batch_text = ""
                for j, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        logger.info(f"Failed to process page {i + j + 1}: {result}")
                        # Retry logic with shorter delay
                        for retry in range(3):
                            logger.info(f"Retrying page {i + j + 1}, attempt {retry + 1}/3")
                            await asyncio.sleep(0.5)
                            try:
                                retry_result = await extract_with_semaphore(batch[j])
                                if retry_result:
                                    batch_text += retry_result
                                    logger.info(f"Retry successful for page {i + j + 1}")
                                    break
                            except Exception as retry_e:
                                if retry == 2:
                                    logger.info(f"All retries failed for page {i + j + 1}: {retry_e}")
                                continue
                    else:
                        # Only add if result is not empty/None
                        if result:
                            batch_text += result

                all_text += batch_text
                logger.info(f"Batch {i // batch_size + 1}/{(len(all_images) + batch_size - 1) // batch_size} completed")

            except Exception as e:
                logger.error(f"Error in batch {i // batch_size + 1}: {e}")
                continue

        return all_text
    except Exception as e:
        logger.error(f"Error in pdf extraction: {e}")
        raise e


async def extract_text_from_file(file_path: str):
    """Extract text from a file based on its extension."""
    file_extension = os.path.splitext(file_path)[1].lower()

    try:
        if file_extension == ".pdf":
            logger.info(f"Extracting text from PDF: {file_path}")
            return await pdf_text_extractor(pdf_path=file_path)
        elif file_extension == ".docx":
            logger.info(f"Extracting text from DOCX: {file_path}")
            return await extract_text_from_docx(file_path=file_path)
        elif file_extension in [".xlsx", ".xls"]:
            logger.info(f"Extracting text from Excel: {file_path}")
            return await extract_text_from_excel_with_llama(file_path=file_path)
        else:
            raise ValueError(f"Unsupported file type: '{file_extension}'")
    except Exception as e:
        logger.error(f"Error extracting text from {file_path}: {e}")
        raise e


# --- Main Multi-Doc Query Functions ---

async def multi_doc_answer(doc_data: str, query: str, doc_name: str):
    """
    Process a single document with the given query and return the answer.
    """
    try:
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)

        system_prompt = multi_doc_query_system_prompt(query=query, doc_data=doc_data)
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{query}"}
            ],
            response_format={"type": "json_object"},
        )

        parsed_json = json.loads(response.choices[0].message.content)

        answer_dict = {
            "doc_name": doc_name,
            "doc_answer": parsed_json.get('answer', 'No answer found')
        }

        return answer_dict

    except Exception as e:
        logger.error(f"Error processing document {doc_name}: {e}")
        return {
            "doc_name": doc_name,
            "doc_answer": f"Error processing document: {str(e)}"
        }


async def multi_doc_query(files: list, query: str, progress_callback=None):
    """
    Process multiple documents (PDF, DOCX, XLSX) with a query and return answers from each.

    Args:
        files: List of tuples (file_path, file_name)
        query: The question to ask about the documents
        progress_callback: Optional callback for progress updates

    Returns:
        Dictionary with results
    """
    try:
        if progress_callback:
            progress_callback(f"Starting multi-document query...")

        # Extract text from all files
        all_docs_data = []
        doc_names = []

        for i, (file_path, file_name) in enumerate(files):
            if progress_callback:
                progress_callback(f"Extracting text from document {i + 1}/{len(files)}: {file_name}")

            try:
                # Extract text based on file type
                content = await extract_text_from_file(file_path)

                if not content or not content.strip():
                    logger.warning(f"No text extracted from {file_name}")
                    content = "No text content could be extracted from this document."

                all_docs_data.append(content)
                doc_names.append(file_name)

            except Exception as e:
                logger.error(f"Error processing {file_name}: {e}")
                # Add error message as content
                all_docs_data.append(f"Error extracting text: {str(e)}")
                doc_names.append(file_name)

        if progress_callback:
            progress_callback(f"Processing query across {len(files)} documents...")

        # Create tasks for parallel processing
        all_tasks = []
        for i, doc_data in enumerate(all_docs_data):
            task = asyncio.create_task(
                multi_doc_answer(
                    doc_data=doc_data,
                    query=query,
                    doc_name=doc_names[i]
                )
            )
            all_tasks.append(task)

        # Wait for all tasks to complete
        answers = await asyncio.gather(*all_tasks)

        if progress_callback:
            progress_callback("Query processing complete!")

        # Format results
        results = {
            'query': query,
            'answers': answers,
            'num_documents': len(files)
        }

        return results

    except Exception as e:
        logger.error(f"Error in multi_doc_query: {e}")
        raise e