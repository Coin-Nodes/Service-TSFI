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
        NSTW Model: Detects Explicit or Implicit Pornography in an Image
        It returns a value from 0.99 to 0.01. Being 0.99 being safe (99% safe)
        and 0.01 being unsafe (1% safe)

        Typically, the values indicate the following possible outcomes:

        => 0.99 would be an image without any kind of nudity captured
            =<0.90 person without pants or shirt, or showing some part of the body, but not pornographic.
            =< 0.80 person without pants or shirt, or showing some part of the body.
            =< 0.20 person without pants and without shirt
            =< 0.10 explicit pornographic content detected.

        Acceptable values to proceed: 0.95 - 0.99
    """
    try:
        image_bytes = await file.read()
        result = CRUDNstw.detect_nsfw(image_bytes)
        return result
    except Exception as e:
        return {"error": str(e)}


model = predict.load_model("/app/nsfw_model/nsfw_mobilenet2.224x224.h5")


@router.post("/nsfw/")
async def detect_nsfw_labels(file: UploadFile = File(...)):
    """
    NSFW Model: Detects Explicit or Implicit Pornography with a More Detailed Index
    Faster model, but may be less sensitive to soft porn detection

    Categories detected:
        drawings: 0.0057229697704315186,
        "hentai": 0.0005879989475943148,
        "neutral": 0.3628992736339569,
        "porn": 0.011622844263911247,
        sexy: 0.619166910648346

    """
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tmp_path = "/tmp/tmp.jpg"
        image.save(tmp_path)
        result = predict.classify(model, tmp_path)
        return result
    except Exception as e:
        return {"error": str(e)}