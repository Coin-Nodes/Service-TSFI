from fastapi import APIRouter
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from starlette.responses import RedirectResponse
import io


from core.use_case.oracle.crud import CRUDOracle

TENSORFLOW_URL = "http://tensorflow:8501/v1/models/rfcn:predict"

router = APIRouter()


@router.get("/hello_oracle")
def hello_oracle():
    return {"hello": "world"}


@router.post("/get_predictions/")
async def get_predictions(file: UploadFile = File(...)):
    """
    Submit a picture to be analyzed by the R-FCN model and retrieve the objects
    identified on the scene.
    """
    image_file = await file.read()
    r = CRUDOracle.get_predictions(image_file, TENSORFLOW_URL)

    return r


@router.post("/get_predicted_image/")
async def get_predicted_image(file: UploadFile = File(...), detections_limit: int = 20):
    """
    Submit a picture to be analyzed by the R-FCN model and get then download the picture
    with all the objects identified in the scene highlighted.

    - **detections_limit** [int]: Define the limit of objects to be ploted in the returning image.
    """
    try:
        image_file = await file.read()
        processed_image = CRUDOracle.get_predicted_image(image_file, TENSORFLOW_URL, detections_limit)
        filename = file.filename.split(".")[0]
        extension = file.filename.split(".")[-1]
        filename = "".join([filename, " (processed).", extension])

        return StreamingResponse(io.BytesIO(processed_image),
                                 headers={'Content-Disposition': 'attachment; filename=' + filename},
                                 media_type="image/jpg")

    except Exception as e:
        return {"error": str(e)}
