import os
import asyncio
import tempfile
import glob
from pathlib import Path
import aioboto3
#from openai import AsyncOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_cohere import CohereRerank

from google import genai
from google.genai import types

from langchain.retrievers import ContextualCompressionRetriever
from constants import AWS_REGION, S3_BUCKET_NAME, OPENAI_EMBEDDING_MODEL,OPENAI_LLM,COHERE_RERANKER_MODEL,FAISS_INDEX_PATH
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
COHERE_API_KEY=os.getenv("COHERE_API_KEY")
AWS_ACCESS_KEY_ID=os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY=os.getenv("AWS_SECRET_ACCESS_KEY")
GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model=OPENAI_EMBEDDING_MODEL)
index = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

get_law_summary_decleration = {
    "name": "get_law_summary",
    "description": " Respond to queries regarding legal or law matters.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Query of the user",
            },
            "language": {
                "type": "string",
                "description": "Language in which answer should be in based on language of user query",
            }
        },
        "required": ["query", "language"],
    },
}

async def get_law_summary(query: str,language: str):
    """
    Main method to get a summarized answer for a legal query.
    """
    retriever = index.as_retriever(search_kwargs={"k": 10})
    compressor = CohereRerank(cohere_api_key=COHERE_API_KEY, model=COHERE_RERANKER_MODEL)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=retriever
    )
    reranked_results = compression_retriever.invoke(query,kwargs=3)
    reranked_sources = [item.metadata['source'] for item in reranked_results]

    if not reranked_sources:
        return "I couldn't find any relevant legal documents to answer your question.", []

    if reranked_results:
        numbers = [f.split('/')[1].split('.')[0] for f in reranked_sources]
        sources=[]
        for item in numbers:
            url = f"https://www.law.go.kr/DRF/lawService.do?OC=doaz&target=law&MST={item}&type=HTML"
            sources.append(url)
        with tempfile.TemporaryDirectory() as temp_dir:
            all_texts = await _download_and_read_sources(reranked_sources, temp_dir)
            return all_texts, sources
            #summary = await _get_openai_summary(query, all_texts,language)
            #return summary, sources

async def _download_and_read_sources( sources: list, temp_dir: str):
    """
    Downloads source files from S3 and reads their content.
    """
    session = aioboto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )

    async def _download_file(s3_client, source):
        try:
            file_path = Path(temp_dir) / Path(source).name
            await s3_client.download_file(
                Bucket=S3_BUCKET_NAME,
                Key=source,
                Filename=str(file_path)
            )
            print(f"{source} ✅ Downloaded successfully!")
        except Exception as e:
            print(f"Failed to download {source}: {e}")

    async with session.client('s3') as s3:
        tasks = [_download_file(s3, source) for source in sources]
        await asyncio.gather(*tasks)

    txt_files = glob.glob(f"{temp_dir}/*.txt")
    all_texts = []
    for file_path in txt_files:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
            all_texts.append(content)

    return all_texts

# async def _get_openai_summary(query: str, all_texts: list,language: str):
#     """
#     Generates a summary using the OpenAI API.
#     """
#     if not all_texts:
#         return "Could not retrieve the content of the relevant legal documents.", []
#     client=AsyncOpenAI(api_key=OPENAI_API_KEY)
#     response = await client.chat.completions.create(
#         model=OPENAI_LLM,
#         messages=[
#             {"role": "system", "content": f"You are a highly intelligent Korean legal assistant. Summarize and analyze the following Korean laws to provide clear and accurate answers to user questions. Use plain language, but remain legally accurate. Laws: \n '{all_texts}'. Give response only in {language} language"},
#             {"role": "user", "content": query}
#         ],
#     )
#     return response.choices[0].message.content

def get_summary(query: str,language: str):
    # Configure the client and tools
    client = genai.Client(api_key=GEMINI_API_KEY)
    tools = types.Tool(function_declarations=[get_law_summary_decleration])
    #system_prompt="You are a highly intelligent Korean legal assistant. Summarize and analyze the following Korean laws to provide clear and accurate answers to user questions. Use plain language, but remain legally accurate. Laws: \n '{all_texts}'. Give response only in {language} language"
    config = types.GenerateContentConfig(tools=[tools])
    contents = [
        types.Content(
            role="user", parts=[types.Part(text=query)]
        )
    ]

    # Send request with function declarations
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=config,
    )

    # Check for a function call
    if response.candidates[0].content.parts[0].function_call:
        function_call = response.candidates[0].content.parts[0].function_call
        if function_call.name == "get_law_summary":
            all_texts,sources = asyncio.run(get_law_summary(**function_call.args))
            function_response_part = types.Part.from_function_response(
                name=function_call.name,
                response={"result": all_texts},
            )

            # Append function call and result of the function execution to contents
            contents.append(response.candidates[0].content)  # Append the content from the model's response.
            contents.append(types.Content(role="user", parts=[function_response_part]))  # Append the function response
            system_prompt = f"You are a highly intelligent Korean legal assistant. Summarize and analyze the following Korean laws to provide clear and accurate answers to user questions. Use plain language, but remain legally accurate. Laws: \n '{all_texts}'. Give response only in {language} language"
            config = types.GenerateContentConfig(tools=[tools], system_instruction=system_prompt)
            final_response = client.models.generate_content(
                model="gemini-2.5-pro",
                config=config,
                contents=contents,
            )

            return final_response.text , sources
    else:
        return response.text, []