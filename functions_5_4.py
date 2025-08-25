import asyncio
import json
import os, io
import pandas as pd
import glob
import fitz
import base64
import tempfile
from pathlib import Path
import logging

from datetime import datetime
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from typing import Dict, Tuple, List, Optional
import concurrent.futures

from PIL import Image
from concurrent.futures import ThreadPoolExecutor

from openai import AsyncOpenAI
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class AsyncExcelWriter:
    """Generate formatted Excel output files asynchronously"""

    def __init__(self, max_workers: int = None):
        """
        Initialize AsyncExcelWriter

        Args:
            max_workers: Maximum number of worker threads for CPU-bound operations
        """
        self.max_workers = max_workers
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    async def write_excel(self, df: pd.DataFrame, output_path, stats: Dict = None) -> str:
        """
        Write DataFrame to formatted Excel file asynchronously

        Args:
            df: DataFrame to write
            output_path: Output file path
            stats: Optional statistics dictionary

        Returns:
            Path to created Excel file
        """
        try:
            logger.info(f"Writing Excel file to: {output_path}")

            # Remove internal columns for output
            output_df = await self._prepare_dataframe(df)

            # Write Excel file in thread pool to avoid blocking
            result_path = await asyncio.get_event_loop().run_in_executor(
                self.executor, self._write_excel_sync, output_df, output_path, stats
            )

            logger.info(f"Excel file created successfully: {result_path}")
            return result_path
        except Exception as e:
            logger.error(f"Error in write_excel: {e}")
            raise e

    async def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare DataFrame by removing internal columns asynchronously"""

        def prepare_sync():
            output_df = df.copy()
            internal_cols = [col for col in output_df.columns if col.startswith('_')]
            return output_df.drop(columns=internal_cols, errors='ignore')

        return await asyncio.get_event_loop().run_in_executor(
            self.executor, prepare_sync
        )

    def _write_excel_sync(self, df: pd.DataFrame, output_path: str, stats: Dict = None) -> str:
        """Synchronous Excel writing (runs in thread pool)"""
        # Create Excel writer
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Write main data
            df.to_excel(writer, sheet_name='Extracted Data', index=False)

            # Add summary sheet if stats provided
            if stats:
                self._write_summary_sheet_sync(writer, stats)

            # Get workbook for formatting
            workbook = writer.book

            # Format main data sheet
            self._format_data_sheet_sync(workbook['Extracted Data'], df)

            # Format summary sheet if exists
            if 'Summary' in workbook.sheetnames:
                self._format_summary_sheet_sync(workbook['Summary'])

        return output_path

    def _write_summary_sheet_sync(self, writer, stats: Dict):
        """Write summary statistics to a separate sheet (synchronous)"""
        summary_data = []

        # Overall statistics
        summary_data.append(['Extraction Summary', ''])
        summary_data.append(['Generated', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
        summary_data.append(['Total Rows Extracted', stats.get('total_rows', 0)])
        summary_data.append(['', ''])

        # By status breakdown
        summary_data.append(['Breakdown by Status', 'Count'])
        for status, count in stats.get('by_status', {}).items():
            summary_data.append([status, count])
        summary_data.append(['', ''])

        # By category breakdown
        summary_data.append(['Breakdown by Category', 'Count'])
        for category, count in stats.get('by_category', {}).items():
            summary_data.append([category, count])

        # Create DataFrame and write
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False, header=False)

    def _format_data_sheet_sync(self, worksheet, df):
        """Apply formatting to the main data sheet with cell merging for '구분' column (synchronous)"""

        # Define styles
        header_font = Font(bold=True, color="FFFFFF")
        header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
        header_alignment = Alignment(horizontal="center", vertical="center")

        border = Border(
            left=Side(style='thin'),
            right=Side(style='thin'),
            top=Side(style='thin'),
            bottom=Side(style='thin')
        )

        # Format headers
        for cell in worksheet[1]:
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = header_alignment
            cell.border = border

        # Find the column index for '구분'
        gubun_col_idx = None
        for idx, col_name in enumerate(df.columns, 1):
            if col_name == '구분':
                gubun_col_idx = idx
                break

        # Merge cells in '구분' column if it exists
        if gubun_col_idx is not None:
            self._merge_gubun_cells_sync(worksheet, df, gubun_col_idx)

        # Format data cells
        for row in worksheet.iter_rows(min_row=2):
            for cell in row:
                cell.border = border
                cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

        # Auto-adjust column widths
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter

            for cell in column:
                try:
                    if cell.value:
                        # Adjust for multi-line content
                        lines = str(cell.value).split('\n')
                        max_line_length = max(len(line) for line in lines)
                        max_length = max(max_length, max_line_length)
                except:
                    pass

            # Set width with min and max bounds
            adjusted_width = min(max(max_length + 2, 10), 50)
            worksheet.column_dimensions[column_letter].width = adjusted_width

        # Freeze top row
        worksheet.freeze_panes = 'A2'

    def _merge_gubun_cells_sync(self, worksheet, df, gubun_col_idx):
        """Merge cells with same values in '구분' column (synchronous)"""
        import pandas as pd

        gubun_values = df['구분'].tolist()
        start_row = 2  # Data starts from row 2 (after header)

        def is_empty_or_nan(value):
            """Check if value is empty, NaN, or None"""
            if pd.isna(value):
                return True
            if value is None:
                return True
            if isinstance(value, str) and value.strip() == '':
                return True
            return False

        def values_equal(val1, val2):
            """Check if two values are equal, considering NaN/empty as equivalent"""
            # Both are empty/NaN
            if is_empty_or_nan(val1) and is_empty_or_nan(val2):
                return True
            # Both are non-empty and equal
            if not is_empty_or_nan(val1) and not is_empty_or_nan(val2):
                return val1 == val2
            # One is empty, one is not
            return False

        i = 0
        while i < len(gubun_values):
            current_value = gubun_values[i]
            start_merge_row = start_row + i
            end_merge_row = start_merge_row

            # Find consecutive rows with same value (including empty/NaN)
            j = i + 1
            while j < len(gubun_values) and values_equal(gubun_values[j], current_value):
                end_merge_row = start_row + j
                j += 1

            # Merge cells if there are multiple rows with same value
            if end_merge_row > start_merge_row:
                # Convert column index to letter
                from openpyxl.utils import get_column_letter
                col_letter = get_column_letter(gubun_col_idx)

                # Merge the cells
                merge_range = f"{col_letter}{start_merge_row}:{col_letter}{end_merge_row}"
                worksheet.merge_cells(merge_range)

                # Set alignment for merged cell
                merged_cell = worksheet[f"{col_letter}{start_merge_row}"]
                merged_cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

                # If the merged cell contains empty/NaN, set it to empty string for display
                if is_empty_or_nan(current_value):
                    merged_cell.value = ""

            i = j

    def _format_summary_sheet_sync(self, worksheet):
        """Apply formatting to the summary sheet (synchronous)"""

        # Define styles
        title_font = Font(bold=True, size=14)
        header_font = Font(bold=True)

        # Format title cells
        for row in worksheet.iter_rows():
            if row[0].value and isinstance(row[0].value, str):
                if 'Summary' in row[0].value or 'Breakdown' in row[0].value:
                    row[0].font = title_font
                elif row[0].value and row[1].value:
                    row[0].font = header_font

        # Auto-adjust column widths
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter

            for cell in column:
                try:
                    if cell.value:
                        max_length = max(max_length, len(str(cell.value)))
                except:
                    pass

            adjusted_width = min(max_length + 2, 50)
            worksheet.column_dimensions[column_letter].width = adjusted_width

    async def close(self):
        """Clean up resources"""
        self.executor.shutdown(wait=True)

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()


def _clean_output(text: str) -> str:
    """Removes common introductory phrases from AI output."""
    prefixes_to_remove = [
        "Here's the extracted text from the image:", "Here is all the text extracted from the image:",
        "Here is the extracted text from the image:", "The extracted text from the image is:",
        "Extracted text:", "Here's the text:", "Here is the text:",
        "Based on the image, here", "From the image, I can see"
    ]
    try:
        cleaned_text = text.strip()
        for prefix in prefixes_to_remove:
            if cleaned_text.lower().startswith(prefix.lower()):
                cleaned_text = cleaned_text[len(prefix):].strip()
                break
        return cleaned_text
    except Exception as e:
        logger.error(f"Error in _clean_output: {e}")
        raise e


async def process_single_page(page_data, temp_dir, total_pages, progress_callback=None):
    """Process a single page to detect tables"""
    semaphore = asyncio.Semaphore(10)
    page_num, img = page_data

    async with semaphore:
        try:
            if progress_callback:
                progress_callback(f"Processing page {page_num + 1}...")

            logger.info(f"Processing page {page_num + 1}...")

            table_detection_prompt = """
            Analyze this image and look for tables that contain these specific Korean column headers:
            - 구분 (Type)
            - 의견사항 (Review Comments)
            - 조치사항 (Actions Taken)
            - 반영여부 (Implementation Status) - should contain values like "반영", "부분반영", or "권고"

            Please respond in JSON format with:
            {
                "has_target_table": true/false,
                "table_count": number_of_matching_tables,
                "table_data": [
                    {
                        "row_number": 1,
                        "구분": "content",
                        "의견사항": "content",
                        "조치사항": "content",
                        "반영여부": "반영/부분반영/권고"
                    }
                ],
                "confidence": "high/medium/low"
            }

            If no matching table is found, set has_target_table to false.
            Extract all rows of data from any matching tables you find.
            Make sure the table contains all 4 required columns: 구분, 의견사항, 조치사항, 반영여부.
            """

            def encode_image_from_pil(pil_image):
                """Encode a PIL Image to base64 string."""
                buffered = io.BytesIO()
                pil_image.save(buffered, format="PNG")
                return base64.b64encode(buffered.getvalue()).decode("utf-8")

            base64_image = encode_image_from_pil(img)
            client = AsyncOpenAI(api_key=OPENAI_API_KEY)
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": table_detection_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", }}
                    ]}
                ],
                response_format={"type": "json_object"}
            )
            response_text = response.choices[0].message.content

            cleaned_response = _clean_output(text=response_text)

            try:
                table_analysis = json.loads(cleaned_response)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON for page {page_num + 1}")
                return None

            # If target table found, save the page image
            if table_analysis.get("has_target_table", False):
                logger.info(f"Found target table on page {page_num + 1}")

                # Calculate zero-padding needed based on total pages
                padding_width = 3

                # Save the page image with zero-padded filename
                page_number_str = str(page_num + 1).zfill(padding_width)
                image_filename = f"page{page_number_str}.png"
                image_path = os.path.join(temp_dir, image_filename)

                # Use thread executor for I/O operations
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    lambda: img.save(image_path, "PNG", quality=95)
                )

                page_result = {
                    "page_number": page_num + 1,
                    "image_saved": image_path,
                    "table_analysis": table_analysis,
                    "table_count": table_analysis.get("table_count", 1)
                }

                logger.info(
                    f"Saved image: {image_filename}, Tables: {table_analysis.get('table_count', 1)}, Confidence: {table_analysis.get('confidence', 'unknown')}")
                return page_result

        except Exception as e:
            logger.error(f"Error processing page {page_num + 1}: {e}")
            return None

    return None


async def detect_and_extract_tables_from_pdf_parallel(pdf_path: str, temp_dir: str, progress_callback=None):
    """Detect and extract tables from PDF pages in parallel"""
    max_concurrent_requests = 30
    batch_size = 30

    try:
        doc = fitz.open(pdf_path)
        os.makedirs(temp_dir, exist_ok=True)

        total_pages = len(doc)
        logger.info(f"Processing {total_pages} pages")

        if progress_callback:
            progress_callback(f"Processing {total_pages} pages...")

        results = {
            "total_pages": total_pages,
            "pages_with_tables": [],
            "table_images_saved": [],
            "extraction_summary": []
        }

        semaphore = asyncio.Semaphore(max_concurrent_requests)

        for batch_start in range(0, total_pages, batch_size):
            batch_end = min(batch_start + batch_size, total_pages)
            batch_pages = list(range(batch_start, batch_end))

            logger.info(f"Processing batch: pages {batch_start + 1}-{batch_end}")
            if progress_callback:
                progress_callback(f"Processing batch: pages {batch_start + 1}-{batch_end}")

            # Extract images directly
            page_images = []
            for page_num in batch_pages:
                page = doc.load_page(page_num)
                pix = page.get_pixmap(dpi=150, alpha=False)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                # Reduce size if needed
                max_size = (1500, 1500)
                if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                    img.thumbnail(max_size, Image.Resampling.LANCZOS)

                page_images.append((page_num, img))
                pix = None  # Clean up pixmap

            # Process pages with semaphore
            async def process_with_semaphore(page_data):
                async with semaphore:
                    return await process_single_page(page_data, temp_dir, total_pages, progress_callback)

            tasks = [process_with_semaphore(page_data) for page_data in page_images]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Clean up images immediately
            for _, img in page_images:
                img.close()
            page_images = None

            # Process results
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.warning(f"Task failed: {result}")
                elif result is not None:
                    results["pages_with_tables"].append(result["page_number"])
                    results["table_images_saved"].append(result["image_saved"])
                    results["extraction_summary"].append(result)

            # Force garbage collection
            import gc
            gc.collect()

            # Delay between batches
            if batch_end < total_pages:
                await asyncio.sleep(1)

        doc.close()

        # Save summary
        summary_path = os.path.join(temp_dir, "extraction_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info("Extraction Complete!")
        logger.info(f"Total pages processed: {results['total_pages']}")
        logger.info(f"Pages with target tables: {len(results['pages_with_tables'])}")
        logger.info(f"Table images saved: {len(results['table_images_saved'])}")

        return results

    except Exception as e:
        logger.error(f"Error in parallel extraction: {e}")
        raise e


async def process_table_images(temp_dir: str, progress_callback=None) -> Tuple[pd.DataFrame, List[Dict]]:
    """Process all PNG images and extract filtered table data with retry logic"""
    max_retries = 3
    # Get all PNG files
    png_files = glob.glob(f"{temp_dir}/*.png")
    logger.info(f"Found {len(png_files)} PNG files to process")

    if progress_callback:
        progress_callback(f"Processing {len(png_files)} table images...")

    client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    # Prompt for table extraction with filtering
    prompt = """
    Extract ALL rows from the Korean table in this image.
    The table has columns: 구분, 의견사항, 조치사항, 반영여부

    Return JSON format:
    {
        "rows": [
            {
                "구분": "content",
                "의견사항": "content",
                "조치사항": "content",
                "반영여부": "반영/부분반영/권고/미반영"
            }
        ]
    }

    Extract every single row, including headers if present.
    """

    all_rows = []
    semaphore = asyncio.Semaphore(10)

    # Track progress
    processed_count = 0
    failed_files = []

    async def process_image_with_retry(img_path_retry, img_index):
        """Process a single image with retry logic"""
        try:
            for attempt in range(max_retries):
                try:
                    async with semaphore:
                        # Log current processing
                        logger.info(
                            f"[{img_index + 1}/{len(png_files)}] Processing: {img_path_retry} (Attempt {attempt + 1}/{max_retries})")

                        if progress_callback:
                            progress_callback(f"Processing image {img_index + 1}/{len(png_files)}")

                        def encode_image(image_path):
                            with open(image_path, "rb") as image_file:
                                return base64.b64encode(image_file.read()).decode("utf-8")

                        base64_image = encode_image(img_path_retry)
                        response = await client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "user", "content": [
                                    {"type": "text", "text": prompt},
                                    {"type": "image_url",
                                     "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", }, }
                                ]}
                            ],
                            response_format={"type": "json_object"}
                        )
                        data = json.loads(response.choices[0].message.content)
                        rows_retry = data.get("rows", [])

                        # Success log
                        logger.info(
                            f"[{img_index + 1}/{len(png_files)}] Successfully processed: {img_path_retry} - Found {len(rows_retry)} rows")
                        return rows_retry, img_index

                except Exception as e:
                    error_msg = f"[{img_index + 1}/{len(png_files)}] Error processing {img_path_retry} (Attempt {attempt + 1}/{max_retries}): {e}"
                    logger.error(error_msg)

                    if attempt < max_retries - 1:
                        # Wait before retry (exponential backoff)
                        wait_time = 2 ** attempt
                        logger.info(f"Retrying in {wait_time} seconds...")
                        await asyncio.sleep(wait_time)
                    else:
                        # Final failure
                        logger.error(f"Failed after {max_retries} attempts for {img_path_retry}")
                        failed_files.append(img_path_retry)
                        return [], img_index  # Return empty list with index

        except Exception as e:
            logger.error(f"Unexpected error in process_image_with_retry: {e}")
            raise e

    # Process all images with index for progress tracking
    tasks = []
    try:
        for idx, img_path in enumerate(png_files):
            task = process_image_with_retry(img_path, idx)
            tasks.append(task)

        # Process all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and sort by index
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Task returned exception: {result}")
            else:
                valid_results.append(result)

        # Sort by index to maintain page order
        sorted_results = sorted(valid_results, key=lambda x: x[1])

        # Combine all rows in correct order
        for rows, img_index in sorted_results:
            if rows:  # Only process if rows is not empty
                all_rows.extend(rows)
                processed_count += 1

        # Filter rows - keep only 반영, 부분반영, 권고
        accepted_statuses = ["반영", "부분반영", "권고"]
        filtered_rows = [
            row for row in all_rows
            if row.get("반영여부", "").strip() in accepted_statuses
        ]

        # Log summary
        logger.info("=" * 60)
        logger.info("PROCESSING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total files processed: {processed_count}/{len(png_files)}")
        logger.info(f"Failed files: {len(failed_files)}")
        if failed_files:
            logger.info("Failed files list:")
            for failed_file in failed_files:
                logger.info(f"  - {failed_file}")
        logger.info(f"Total rows extracted: {len(all_rows)}")
        logger.info(f"Filtered rows (반영/부분반영/권고): {len(filtered_rows)}")
        logger.info("=" * 60)

        # Create DataFrame
        df = pd.DataFrame(filtered_rows)

        return df, filtered_rows
    except Exception as e:
        logger.error(f"Error during creating dataframe: {e}")
        raise e


async def extraction_review(pdf_file_path: str, progress_callback=None) -> Dict:
    """
    Main function to extract review data from PDF

    Args:
        pdf_file_path: Path to the PDF file
        progress_callback: Optional callback function for progress updates

    Returns:
        Dictionary containing:
        - df: pandas DataFrame with extracted data
        - json_data: List of dictionaries with extracted data
        - excel_file_path: Path to the generated Excel file
        - summary: Processing summary
    """
    temp_dir = None
    try:
        # Create a unique temp directory
        temp_dir = tempfile.mkdtemp(prefix=f"extraction_")
        logger.info(f"Created temporary directory: {temp_dir}")

        if progress_callback:
            progress_callback("Starting PDF processing...")

        file_name, file_extension = os.path.splitext(os.path.basename(pdf_file_path))

        if file_extension.lower() != ".pdf":
            raise ValueError(f"Wrong file input: expected .pdf, got {file_extension}")

        logger.info("Starting PDF table extraction")
        if progress_callback:
            progress_callback("Detecting tables in PDF pages...")

        extracted_text = await detect_and_extract_tables_from_pdf_parallel(
            pdf_path=pdf_file_path,
            temp_dir=temp_dir,
            progress_callback=progress_callback
        )

        logger.info("Starting table image processing")
        if progress_callback:
            progress_callback("Extracting data from detected tables...")

        df, json_data = await process_table_images(temp_dir=temp_dir, progress_callback=progress_callback)

        # Create output files
        json_file_path = Path(temp_dir) / "extracted_data.json"
        excel_file_path = Path(temp_dir) / "extracted_data.xlsx"

        if progress_callback:
            progress_callback("Creating output files...")

        # Save JSON data
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)

        # Create Excel file with formatting
        excel_writer = AsyncExcelWriter()
        merged_excel_file_path = await excel_writer.write_excel(
            df=df,
            output_path=excel_file_path
        )
        await excel_writer.close()

        logger.info(f"Created Excel file: {merged_excel_file_path}")

        # Create summary
        summary = {
            "total_pages": extracted_text.get("total_pages", 0),
            "pages_with_tables": len(extracted_text.get("pages_with_tables", [])),
            "total_rows_extracted": len(json_data),
            "filtered_rows": len(df) if not df.empty else 0,
            "processing_status": "completed"
        }

        if progress_callback:
            progress_callback("Extraction completed successfully!")

        logger.info("Extraction review completed successfully")

        return {
            "df": df,
            "json_data": json_data,
            "excel_file_path": str(merged_excel_file_path),
            "json_file_path": str(json_file_path),
            "summary": summary,
            "temp_dir": temp_dir  # Return temp_dir so caller can clean it up if needed
        }

    except Exception as e:
        logger.error(f"Error in extraction_review: {e}")
        # Clean up on error
        if temp_dir and os.path.exists(temp_dir):
            try:
                import shutil
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up temp directory: {cleanup_error}")
        raise e


def cleanup_temp_directory(temp_dir: str):
    """Helper function to clean up temporary directory"""
    if temp_dir and os.path.exists(temp_dir):
        try:
            import shutil
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up temp directory: {e}")