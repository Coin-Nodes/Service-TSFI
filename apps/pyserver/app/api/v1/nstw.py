from fastapi import APIRouter
from fastapi import APIRouter, UploadFile, File
from nsfw_detector import predict
from core.use_case.nstw.crud import CRUDNstw
from PIL import Image
import io

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


# model = predict.load_model("/app/nsfw_model/nsfw_mobilenet2.224x224.h5")

def detect_nsfw_labels(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tmp_path = "/tmp/tmp.jpg"
    image.save(tmp_path)
    result = predict.classify(model, tmp_path)
    return result