import logging
import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from langchain.agents import AgentType, initialize_agent
from langchain_aws import BedrockLLM

from tools import (
    LLM_NAME,
    bedrock_client,
    extract_text_from_s3,
    summarize_text_with_bedrock,
    upload_file_to_s3,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(filename)s:%(lineno)s - %(name)s - %(funcName)s - %(levelname)s - %(message)s",
)


# Definning the FastAPI App
app = FastAPI(title="Agentic Intelligent Document Summarizer")


# Definning the Agent
def create_agent():
    try:
        tools = [upload_file_to_s3, extract_text_from_s3, summarize_text_with_bedrock]
        agent = initialize_agent(
            tools=tools,
            llm=BedrockLLM(client=bedrock_client, model_id=LLM_NAME),
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
        )
        return agent
    except Exception as e:
        logging.error(f"Exception occurred while running the agent: {str(e)}")
        raise RuntimeError(f"Failed to run the agent: {str(e)}")


# Definning the Route
@app.post("/")
async def process_document(file: UploadFile = File(...)):
    # Save file locally so that it can be uploaded to S3
    logging.info("Executing the main funciton")
    file_path = f"/media_files/{file.filename}"
    try:
        if not os.path.exists("/media_files"):
            os.makedirs("/media_files")
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Define S3 object name
        object_name = file.filename

        # Create and run the agent
        agent = create_agent()
        result = agent.invoke(
            {
                "input": f"Upload the file '{file_path}' as '{object_name}' in AWS S3, then extract text from the uploaded file using AWS Textract, and finally summarize the text extracted using BedRock."
            }
        )

        # Clean up local file
        os.remove(file_path)

        return JSONResponse(content={"summary": result["output"]}, status_code=200)
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
