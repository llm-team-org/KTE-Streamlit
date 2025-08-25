import os
import time
import asyncio
import fitz
import glob
from PIL import Image
import tempfile
import json
import pandas as pd
from pathlib import Path
import logging

from openai import AsyncOpenAI
from google import genai
from google.genai import types

from constants import OPENAI_LLM

from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

finishing_material_drawing_system_prompt="""
    Extract ALL rows from the Korean table in this image.
    The table has columns:

     - 구 분 (Type)
     - 실 명 (Name)
     - 바 닥 바 탕 (floor outline)
     - 바 닥 마 감 (floor finishing)
     - 벽 체 바 탕 (wall outline)
     - 벽 체 마 감 (wall finishing)
     - 벽 체 걸레받이 (wall baseboard)
     - 천장 바 탕 (ceiling outline)
     - 천장 마 감 (ceiling finishing)

2. **Data Handling**
   - Extract every row in the table.
   - Preserve text exactly as it appears (do not translate or modify).
   - If a cell is empty, return it as an empty string (`""`).

## Example JSON Output
[
  {
    "구 분 (Type)": "구분 값",
    "실 명 (Name)": "방 이름",
    "바 닥 바 탕 (floor outline)": "값",
    "바 닥 마 감 (floor finishing)": "값",
    "벽 체 바 탕 (wall outline)": "값",
    "벽 체 마 감 (wall finishing)": "값",
    "벽 체 걸레받이 (wall baseboard)": "값",
    "천장 바 탕 (ceiling outline)": "값",
    "천장 마 감 (ceiling finishing)": "값"
  }
]
"""

finishing_material_contract_system_prompt="""
You are a expert construction specification contractor.

For the given image containing a table, carefully read and extract all contract information following these rules:

Image structure is very inconsitant so you need to extra careful for the extraction of data.

Colums names in table will not be specified or even if colums names are specified just ignore them and follow the following structure colum wise:

In the table image you to extract these 구 분 (Type), 실 명 (Name), 객실 요소 (Room Elements) and 마감기준 (Materiasl)
To extract these you need to firstly carefully read the image because in some images you will see 4 colums in some you will 3 colums thats where
your intelligence will come in even if there are 3 colums or 4 colums you need to intelligently see when data is for which case.

like for 객실 요소 (Room Elements) will have data like 바 닥 (Floor), 벽 체 (Wall) or 벽 체 걸레받이 (wall baseboard),걸레받이(mop holder) and 천장 (Ceiling) etc.
마감기준 (Materiasl) will have materials used in for each 객실 요소 (Room Elements)
실 명 (Name) will have room names or area names like kitchen, sauna, elevator or elev, bedroom , TV lounge, living room etc.
구 분 (Type) is the most hardest to find so be very extracarful for this, it can includes 전 용 부위 (Exclusive Area) ,공 용 부위 (Public Area), 각종설계기준 (Various design standards), 지상 /지하 주차장(Ground/underground parking lot), 전기/발전기실/열교환기실 (Electrical/generator room/heat exchanger room),기계실 (machine room),우수저류조 (Rainwater storage tank),어린이집(daycare center), 경로당(Senior Citizens' Party), 관리사무소(Management Office) and many more etc will come into this 구 분 (Type).


In some cases you will get images like this:

1st colum of the table will always be 구 분 (Type),
2nd colum of the table will always be 실 명 (Name),
3rd colum of the table will always be 객실 요소 (Room Elements) i.e 바 닥 (Floor), 벽 체 (Wall) and 천장 (Ceiling) etc,
4th colum of the table will always be 마감기준 (Materiasl).

In some cases you will get images likes this

1st colum of the table will always be 구 분 (Type) and 실 명 (Name) combined,
2nd colum of the table will always be 객실 요소 (Room Elements) i.e 바 닥 (Floor), 벽 체 (Wall) and 천장 (Ceiling) etc,
3rd colum of the table will always be 마감기준 (Materiasl).

and in some cases you will get images like

1st colum of the table will always be 구 분 (Type),
2nd colum of the table will always be 실 명 (Name) and 객실 요소 (Room Elements) combined,
3rd colum of the table will always be 마감기준 (Materiasl).

Your task:
1. Carefully read the image and get 구 분 (Type),실 명 (Name), 객실 요소 (Room Elements) and 마감기준 (Materials) from table.

2. **Merged Cell Handling:**
   - Firstly you need to carefully read the whole table image first because 구 분 (Type) and 실 명 (Name) cells can be merged.
      so if you find something their you need put their that unless you find closing row line in table.
      This is the most important and complex data to extract so be extra carefull for 구 분 (Type) and 실 명 (Name)
   - When 구 분 (Type) and 실 명 (Name) spans multiple rows, apply that same value to ALL rows until the next category appears
   - Example: If "주방" spans 8 rows, use "주방" for all 8 entries

3. **Empty Cell Handling:**
   - If any cell is empty or missing content in any row, use empty string ""
   - Do not skip rows with empty cells - include them in the output

4. **Flexible Structure:**
   - Ignore actual header names - focus on content types
   - Table may have 3+ columns, but extract only the 3 content types above
   - Read all rows systematically regardless of table layout

Return ONLY in JSON format:
[
    {
        "구 분 (Type)": "content or empty string",
        "실 명 (Name)": "content or empty string",
        "객실 요소 (Room Elements)": "content or empty string",
        "마감기준 (Materials)": "content or empty string"
    }
]

Extract every single row including those with empty cells, ensuring merged category values are repeated for each corresponding entry.
"""

def finishing_material_system_prompt(drawing_json,contract_json):
    return """You are a construction material comparison assistant. Your task is to compare materials between two JSON files: a Drawing specification and a Contract specification.

## Input Files Structure:

**Drawing JSON** contains:
- 구 분 (Type): Division/category
- 실 명 (Name): Room/area name
- Material columns: 바 닥 마 감 (floor finishing), 벽 체 마 감 (wall finishing), 천장 마 감 (ceiling finishing), etc.

**Contract JSON** contains:
- 구 분 (Type): Division/category
- 실 명 (Name): Room/area name  
- 객실 요소 (Room Elements): Specifies element type (바닥/Floor, 벽체/Wall, 천장/Ceiling, etc.)
- 마감기준 (Materials): Material specification for that element

## Your Task:
1. For each room in each division, identify all materials specified
2. Match Drawing materials with Contract materials based on element type
3. Create comparison entries showing both specifications

## Matching Rules:
- Match rooms by 실 명 (Name) within same 구 분 (Type/Division)
- Floor materials: 바 닥 마 감 in Drawing ↔ 바닥 in Contract
- Wall materials: 벽 체 마 감 in Drawing ↔ 벽체 in Contract  
- Ceiling materials: 천장 마 감 in Drawing ↔ 천장 in Contract
- Other elements should be matched by their Korean terms

## Output Format:
Return a JSON array where each entry represents one element comparison:
object name always will be 'result'
{
  "구 분": "division name",
  "실 명": "room name",
  "element": "element type (floor/wall/ceiling)",
  "Drawing": "material from drawing or empty string",
  "Contract": "material from contract or empty string"
}
"""

async def find_and_save_pages_as_images(pdf_path: str, temp_dir: str):
    """Find pages with '실내재료 마감표' keyword and save them as images"""
    dpi = 150
    keyword = "실내재료 마감표"
    start_time = time.time()
    output_dir = os.path.join(temp_dir, "tables")
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Scanning PDF for '{keyword}'...")
    try:
        doc = fitz.open(pdf_path)

        # Calculate zoom factor from DPI
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)

        found_pages = []
        saved_images = []

        # Scan and save pages
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()

            if keyword in text:
                found_pages.append(page_num + 1)

                # Immediately save this page as image
                logger.info(f"Found keyword on page {page_num + 1}, saving as image...")

                # Render page to image
                pix = page.get_pixmap(matrix=mat)

                # Save as PNG
                image_path = f"{output_dir}/page_{page_num + 1}.png"
                pix.save(image_path)
                saved_images.append(image_path)

                logger.info(f"✓ Saved: {image_path}")

        doc.close()

        # Summary
        elapsed = time.time() - start_time
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Scan Complete! for Drawing pdf")
        logger.info(f"Found {len(found_pages)} pages with '{keyword}': {found_pages}")
        logger.info(f"Saved {len(saved_images)} images in: {output_dir}/")
        logger.info(f"Time taken: {elapsed:.2f} seconds")
        logger.info(f"{'=' * 50}")

        return saved_images
    except Exception as e:
        logger.error(f"Error in find_and_save_pages_as_images: {e}")
        raise e


async def drawing_pdf_content(images_path: list):
    """Extract content from drawing PDF images using Gemini"""
    try:
        all_images = []
        for img in images_path:
            all_images.append(Image.open(img))

        client = genai.Client(api_key=GEMINI_API_KEY)

        async def get_response(img):
            retry_count = 0
            max_retries = 3

            while retry_count <= max_retries:
                try:
                    response = await client.aio.models.generate_content(
                        model="gemini-2.5-flash",
                        config=types.GenerateContentConfig(
                            system_instruction=finishing_material_drawing_system_prompt,
                            response_mime_type="application/json"
                        ),
                        contents=[img]
                    )
                    return response.text

                except Exception as e:
                    if "429" in str(e) and retry_count < max_retries:
                        retry_count += 1
                        wait_time = retry_count * 2  # 2, 4, 6 seconds
                        logger.info(f"Rate limit hit, retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    else:
                        raise e

        async def process_images_in_batches(all_images):
            all_json = []
            batch_size = 10
            # Process images in batches
            for i in range(0, len(all_images), batch_size):
                batch = all_images[i:i + batch_size]
                logger.info(
                    f"Processing Drawing batch {i // batch_size + 1}: images {i + 1}-{min(i + batch_size, len(all_images))}")

                # Create tasks for current batch
                batch_tasks = [get_response(img) for img in batch]

                try:
                    # Wait for current batch to complete
                    batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

                    # Handle any exceptions in the batch
                    for j, result in enumerate(batch_results):
                        if isinstance(result, Exception):
                            logger.error(f"Error processing image {i + j + 1}: {result}")
                            all_json.append(None)
                        else:
                            all_json.append(result)
                except Exception as e:
                    logger.error(f"Batch {i // batch_size + 1} failed: {e}")
            return all_json

        drawing_json = await process_images_in_batches(all_images)
        return drawing_json
    except Exception as e:
        logger.error(f"Error in drawing_pdf_content: {e}")
        raise e


async def contract_pdf_content(pdf_path: str, temp_dir: str):
    """Extract content from contract PDF using Gemini"""
    try:
        doc = fitz.open(pdf_path)
        page_count = len(doc)
        output_dir = os.path.join(temp_dir, "contract")
        os.makedirs(output_dir, exist_ok=True)

        # Convert each page to image with 150 DPI
        for page_num in range(page_count):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=fitz.Matrix(150 / 72, 150 / 72))
            pix.save(f"{temp_dir}/contract/page_{page_num + 1:03d}.png")

        doc.close()
        logger.info(f"Converted {page_count} pages to images with 150 DPI in '{output_dir}' folder")

        image_files = sorted(glob.glob(f"{output_dir}/*.png"))
        images = []
        for image_file in image_files:
            img = Image.open(image_file)
            images.append(img)

        logger.info(f"Total images loaded: {len(images)}")

        client = genai.Client(api_key=GEMINI_API_KEY)

        async def process_image_with_retry(img):
            retry_count = 0
            max_retries = 3

            while retry_count <= max_retries:
                try:
                    response = await client.aio.models.generate_content(
                        model="gemini-2.5-pro",
                        config=types.GenerateContentConfig(
                            system_instruction=finishing_material_contract_system_prompt,
                            response_mime_type="application/json"
                        ),
                        contents=[img]
                    )
                    return response.text

                except Exception as e:
                    if "429" in str(e) and retry_count < max_retries:
                        retry_count += 1
                        wait_time = retry_count * 2  # 2, 4, 6 seconds
                        logger.info(f"Rate limit hit, retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    else:
                        raise e

        async def process_images_with_retry_batched(images):
            all_json = []
            batch_size = 10
            # Process images in batches
            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size]
                logger.info(
                    f"Processing Contract batch {i // batch_size + 1}: images {i + 1}-{min(i + batch_size, len(images))}")

                # Create tasks for current batch
                batch_tasks = [process_image_with_retry(img) for img in batch]

                try:
                    # Wait for current batch to complete
                    batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

                    # Handle any exceptions in the batch
                    for j, result in enumerate(batch_results):
                        if isinstance(result, Exception):
                            logger.error(f"Error processing image {i + j + 1}: {result}")
                            all_json.append(None)
                        else:
                            all_json.append(result)
                except Exception as e:
                    logger.error(f"Batch {i // batch_size + 1} failed: {e}")

            return all_json

        contract_json = await process_images_with_retry_batched(images)
        return contract_json
    except Exception as e:
        logger.error(f"Error in contract_pdf_content: {e}")
        raise e


async def finishing_material_comparison(drawing_pdf_path: str, contract_pdf_path: str, progress_callback=None):
    """Compare drawing and contract PDFs and generate Excel report"""
    temp_dir = None
    temp_dir2 = None

    try:
        temp_dir = tempfile.mkdtemp(prefix=f"finishing_")
        temp_dir2 = tempfile.mkdtemp(prefix=f"finishing_C_")

        # Progress update
        if progress_callback:
            progress_callback("Starting comparison process...")

        # Process drawing PDF
        if progress_callback:
            progress_callback("Processing drawing PDF...")
        table_images_path = await find_and_save_pages_as_images(pdf_path=drawing_pdf_path, temp_dir=temp_dir)
        logger.info("Tables getting success")

        # Get content of drawing PDF
        if progress_callback:
            progress_callback("Extracting drawing content...")
        drawing_json = await drawing_pdf_content(images_path=table_images_path)

        # Process contract PDF
        if progress_callback:
            progress_callback("Processing contract PDF...")

        # Get content of contract PDF
        if progress_callback:
            progress_callback("Extracting contract content...")
        contract_json = await contract_pdf_content(pdf_path=contract_pdf_path, temp_dir=temp_dir)

        # Perform comparison using OpenAI
        if progress_callback:
            progress_callback("Performing AI comparison...")

        try:
            client = AsyncOpenAI(api_key=OPENAI_API_KEY)
            response = await client.chat.completions.create(
                model=OPENAI_LLM,
                messages=[
                    {"role": "system", "content": finishing_material_system_prompt(drawing_json=drawing_json,
                                                                                   contract_json=contract_json)},
                    {"role": "user",
                     "content": f"Kindly do the comparison carefully between these two jsons: Drawing json {drawing_json} and Contract Json {contract_json}"}
                ],
                response_format={"type": "json_object"}
            )
            data = json.loads(response.choices[0].message.content)

            # Create DataFrame and Excel file
            if progress_callback:
                progress_callback("Creating Excel report...")

            df = pd.DataFrame(data['result'])
            excel_file_path = Path(temp_dir) / "finishing_comparison.xlsx"
            df.to_excel(excel_file_path, index=False)

            if progress_callback:
                progress_callback("Comparison complete!")

            return {
                'df': df,
                'excel_file_path': str(excel_file_path),
                'temp_dir': temp_dir,
                'temp_dir2': temp_dir2
            }

        except Exception as e:
            logger.error(f"Error in comparison: {e}")
            raise e

    except Exception as e:
        # Clean up on error
        if temp_dir and os.path.exists(temp_dir):
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except:
                pass
        if temp_dir2 and os.path.exists(temp_dir2):
            try:
                import shutil
                shutil.rmtree(temp_dir2)
            except:
                pass
        raise e


def cleanup_temp_directories_finishing(temp_dir: str, temp_dir2: str = None):
    """Clean up temporary directories"""
    if temp_dir and os.path.exists(temp_dir):
        try:
            import shutil
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up temp directory: {e}")

    if temp_dir2 and os.path.exists(temp_dir2):
        try:
            import shutil
            shutil.rmtree(temp_dir2)
            logger.info(f"Cleaned up temporary directory: {temp_dir2}")
        except Exception as e:
            logger.warning(f"Failed to clean up temp directory: {e}")