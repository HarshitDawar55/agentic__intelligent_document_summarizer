import json
import logging
import os

import boto3
from langchain.tools import tool

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(filename)s:%(lineno)s - %(name)s - %(funcName)s - %(levelname)s - %(message)s",
)

# Definning S3_Bucket_Name
S3_BUCKET = os.getenv("S3_BUCKET")

# Definning the required AWS Clients
s3_client = boto3.client(
    "s3",
    region_name="us-east-1",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)
textract_client = boto3.client(
    "textract",
    region_name="us-east-1",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)
bedrock_client = boto3.client(
    "bedrock-runtime",
    region_name="us-east-1",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)


# Definning AWS LLM
LLM_NAME = "meta.llama3-8b-instruct-v1:0"


@tool
def upload_file_to_s3(prompt: str) -> str:
    """
    This tool is only used to upload a file to AWS S3.
    Input format: "file_path=<file_path>, object_name=<object_name>"
    """
    try:
        logging.info(f"Into the upload_file_to_s3 function with input: {prompt}")
        parts = dict(item.split("=") for item in prompt.split(", "))
        file_path = parts.get("file_path")
        object_name = parts.get("object_name")

        if not file_path or not object_name:
            raise ValueError("Both file_path and object_name must be provided.")

        s3_client.upload_file(file_path, S3_BUCKET, object_name)
        return f"File {object_name} uploaded successfully to S3 bucket {S3_BUCKET}."
    except Exception as e:
        logging.error(f"Exception in uploading the file to S3: {str(e)}")
        raise RuntimeError(f"Failed to upload file: {str(e)}")


@tool
def extract_text_from_s3(prompt: str) -> str:
    """
    This tool is only used to extract text from a file/document stored in S3 using AWS Textract.
    Input format: "bucket_name=<bucket_name>, object_name=<object_name>"
    """
    try:
        logging.info(f"Into the extract_text_from_s3 function with input: {prompt}")
        parts = dict(item.split("=") for item in prompt.split(", "))
        bucket_name = parts.get("bucket_name")
        object_name = parts.get("object_name")

        if not bucket_name or not object_name:
            raise ValueError("Both file_path and object_name must be provided.")

        response = textract_client.start_document_analysis(
            DocumentLocation={
                "S3Object": {
                    "Bucket": bucket_name,
                    "Name": object_name,
                }
            },
            FeatureTypes=["TABLES", "FORMS"],
        )

        job_id = response["JobId"]
        print(f"Job started with ID: {job_id}")

        # Wait for job completion
        while True:
            status = textract_client.get_document_analysis(JobId=job_id)
            if status["JobStatus"] in ["SUCCEEDED", "FAILED"]:
                break
        if status["JobStatus"] == "SUCCEEDED":
            extracted_text = []
            for block in status["Blocks"]:
                if block["BlockType"] == "LINE":
                    extracted_text.append(block["Text"])
            return " ".join(extracted_text)
        else:
            return f"Job Status: {status['JobStatus']}"
    except Exception as e:
        logging.error(f"Exception in extracting text from file: {str(e)}")
        raise RuntimeError(f"Failed to extract text from file: {str(e)}")


@tool
def summarize_text_with_bedrock(text: str) -> str:
    """This tool is only used to summarize text using AWS Bedrock."""
    try:
        logging.info(
            f"Into the summarize_text_with_bedrock function with input: {text}"
        )
        body_request = {
            "prompt": f"As an expert researcher and technical expert, please summarize the following text: {text}",
            "temperature": 0.5,
        }

        response = bedrock_client.invoke_model(
            modelId=LLM_NAME, body=json.dumps(body_request)
        )
        model_response = json.loads(response["body"].read())
        response_text = model_response["generation"].strip()
        return response_text
    except Exception as e:
        logging.error(f"Exception in summarizing the text: {str(e)}")
        raise RuntimeError(f"Failed to summarize the text: {str(e)}")
