from fastapi import FastAPI, HTTPException
import logging
import json
import uvicorn
from pydantic import BaseModel, Field
from typing import Optional
from assistant import Assistant

# Initialize objects
assistant_pi_obj = Assistant()

# Initialize FastAPI
app = FastAPI()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class PineconeRequest(BaseModel):
    session_id: str
    query: str
    category: Optional[str] = Field(default="block_chain")

@app.get("/")
async def read_root():
    logger.info("Root endpoint accessed")
    return {"hello": "3rd-eye_assistant"}

@app.post("/assistant")
async def assistant_response(request: PineconeRequest):
    logger.info("Pinecone response endpoint accessed for session ID: %s", request.session_id)
    try:
        response = assistant_pi_obj.get_final_answer(request.session_id, request.query, request.category)
        logger.info("Successfully generated Pinecone response for session ID: %s", request.session_id)
        return {
            "statusCode": 200,
            "body": response
        }
    except Exception as e:
        logger.error("Error while generating Pinecone response for session ID: %s: %s", request.session_id, e)
        return {
            "statusCode": 400,
            "body": json.dumps(str(e))
        }

@app.get("/")
def read_root():
    return {"message": "FastAPI is running on 52.237.203.110"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="52.237.203.110", port=5900, reload=False)


