import io

from fastapi import APIRouter

TENSORFLOW_URL = "http://tensorflow:8501/v1/models/rfcn:predict"

router = APIRouter()


@router.get("/hello")
def hello():
    return {"hello": "world"}
