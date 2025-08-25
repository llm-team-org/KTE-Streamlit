import os, io
import base64
import fitz
from PIL import Image
import tempfile

from openai import AsyncOpenAI
from constants import OPENAI_LLM
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

tool = [{
    "type": "function",
    "function": {
        "name": "get_image",
        "description": "Gets images data for user query",
        "parameters": {
            "type": "object",
            "properties": {
                "page_no": {
                    "type": "string",
                    "description": "pages of images"
                }
            },
            "required": ["page_no"],
            "additionalProperties": False
        },
        "strict": True
    }
}]


async def get_image(page_no: int, pdf_path):
    """Extract image from a specific page of the PDF."""
    doc = None
    try:
        # Ensure file exists
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        doc = fitz.open(pdf_path)

        # Validate page number
        if page_no < 1 or page_no > len(doc):
            raise ValueError(f"Invalid page number {page_no}. PDF has {len(doc)} pages.")

        page = doc.load_page(page_no - 1)

        # Get pixmap with error handling
        try:
            pix = page.get_pixmap(dpi=150, alpha=False)
        except Exception as e:
            # Try with lower DPI if high DPI fails
            pix = page.get_pixmap(dpi=100, alpha=False)

        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=85)
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return img_str
    except Exception as e:
        raise Exception(f"Error in PDF page extraction: {str(e)}")
    finally:
        if doc:
            doc.close()


async def drawing_chatbot(page_no: int, user_query: str, pdf_file_path: str):
    """
    Main drawing chatbot function that analyzes PDF pages based on user queries.

    Args:
        page_no: Page number to analyze (1-indexed)
        user_query: User's question about the drawing
        pdf_file_path: Path to the PDF file

    Returns:
        tuple: (status, answer) where status indicates success/failure and answer contains the response
    """
    try:
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)

        system_prompt = """
        You are an intelligent Drawing Analysis Chatbot. You will receive an image and a user query about that image.
        Your task is to carefully analyze the image and respond to the user.

        Use the `get_image` tool/function **only** if:
        - The answer to the user's query requires extracting specific details from the image.

        If the user's query can be answered without analyzing the image in detail, do **not** use the tool.

        If the answer to the user's query is not present in the image, clearly respond:
        "This image does not contain the answer to your query."

        Always provide clear, concise, and accurate answers.
        """

        # Initial request
        response = await client.chat.completions.create(
            model=OPENAI_LLM,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            tools=tool,
            tool_choice="auto",
            stream=True
        )

        accumulated_content = ""
        tool_calls_accumulated = []
        current_tool_call = None

        # Process the stream
        async for chunk in response:
            choice = chunk.choices[0]
            delta = choice.delta

            # Handle tool calls
            if delta.tool_calls:
                for tool_call_chunk in delta.tool_calls:
                    if tool_call_chunk.index is not None:
                        # New tool call or updating existing one
                        while len(tool_calls_accumulated) <= tool_call_chunk.index:
                            tool_calls_accumulated.append({
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""}
                            })

                        current_tool_call = tool_calls_accumulated[tool_call_chunk.index]

                        if tool_call_chunk.id:
                            current_tool_call["id"] = str(tool_call_chunk.id)
                        if tool_call_chunk.function:
                            if tool_call_chunk.function.name:
                                current_tool_call["function"]["name"] = tool_call_chunk.function.name
                            if tool_call_chunk.function.arguments:
                                current_tool_call["function"]["arguments"] += tool_call_chunk.function.arguments

            # Handle regular content (when no tool calls)
            if delta.content is not None:
                content = delta.content
                accumulated_content += content

                # Yield intermediate results for streaming
                yield ("streaming", accumulated_content)

        # Check if tool calls were made
        if tool_calls_accumulated and any(tc["function"]["name"] for tc in tool_calls_accumulated):
            # Process tool calls
            image = None
            for tool_call in tool_calls_accumulated:
                function_name = tool_call["function"]["name"]
                if function_name == "get_image":
                    try:
                        image = await get_image(page_no=page_no, pdf_path=pdf_file_path)
                    except Exception as e:
                        yield ("error", f"Failed to extract image: {str(e)}")
                        return

            if image:
                detailed_system_prompt = f"""
                You are an intelligent Drawing Analysis Chatbot. You will receive an image and a user query about that image.
                Your task is to carefully analyze the image and respond to the user.

                Remember: Always respond only in the user's query language. if user asked question in english then response in english"""

                # Build messages history with the detailed system prompt
                messages_history = [
                    {"role": "system", "content": detailed_system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": f"{user_query}"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}", }}
                    ]},
                    {"role": "assistant", "content": None, "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["function"]["name"],
                                "arguments": tc["function"]["arguments"]
                            }
                        } for tc in tool_calls_accumulated
                    ]}
                ]

                # Add tool responses
                for tool_call in tool_calls_accumulated:
                    if tool_call["function"]["name"] == "get_image":
                        messages_history.append({
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "name": tool_call["function"]["name"],
                            "content": str(image)
                        })

                # Send follow-up with streaming
                followup = await client.chat.completions.create(
                    model=OPENAI_LLM,
                    messages=messages_history,
                    temperature=0.2,
                    stream=True,
                )

                text = ""
                async for chunk in followup:
                    if chunk.choices[0].delta.content is not None:
                        content = chunk.choices[0].delta.content
                        text += content
                        yield ("streaming", text)

                yield ("complete", text)
        else:
            # No tool calls - just regular response
            yield ("complete", accumulated_content)

    except Exception as e:
        yield ("error", f"An error occurred: {str(e)}")


def get_pdf_info(pdf_path):
    """Get information about the PDF file."""
    try:
        doc = fitz.open(pdf_path)
        num_pages = len(doc)
        doc.close()
        return {"num_pages": num_pages}
    except Exception as e:
        raise Exception(f"Error reading PDF: {e}")