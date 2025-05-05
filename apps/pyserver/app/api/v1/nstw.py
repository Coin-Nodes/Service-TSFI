from fastapi import APIRouter
from fastapi import APIRouter, UploadFile, File

from core.use_case.nstw.crud import CRUDNstw

TENSORFLOW_URL = "http://tensorflow:8501/v1/models/rfcn:predict"

router = APIRouter()


@router.get("/hello")
def hello():
    return {"hello": "world"}



@router.post("/nstw/")
async def detect_nstw(file: UploadFile = File(...)):
    """
    Detecta se uma imagem contém conteúdo NSFW.
    """
    try:
        image_bytes = await file.read()
        result = CRUDNstw.detect_nsfw(image_bytes)
        return result
    except Exception as e:
        return {"error": str(e)}