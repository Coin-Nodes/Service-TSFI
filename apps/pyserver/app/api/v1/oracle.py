import io

import objectdetection
from fastapi import APIRouter
from fastapi import File, UploadFile
from fastapi.responses import StreamingResponse

TENSORFLOW_URL = "http://tensorflow:8501/v1/models/rfcn:predict"

router = APIRouter()


@router.get("/oi")
def hello():
    return {"hello": "world"}



