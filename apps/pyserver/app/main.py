##################################################
##  Author:
##  Date:
##################################################

import io

import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from starlette.responses import RedirectResponse

import objectdetection
from api.v1 import api_router

TENSORFLOW_URL = "http://tensorflow:8501/v1/models/rfcn:predict"

app = FastAPI(
    title="Trained System for Feature Identification - TSFI",
    description="wwww.coinnodes.tech -allows image uploading for a TensorFlow container running a R-FCN pre-trained model for object identification",
    version="1.0.0",
)

app.include_router(api_router, prefix="/api/v1")


@app.get("/")
def home_screen():
    return RedirectResponse(url='/docs')


@app.post("/get_predictions/")
async def get_predictions(file: UploadFile = File(...)):
    """
    Submit a picture to be analyzed by the R-FCN model and retrieve the objects
    identified on the scene.
    """
    image_file = await file.read()
    r = objectdetection.get_predictions(image_file, TENSORFLOW_URL)

    return r


@app.post("/get_predicted_image/")
async def get_predicted_image(file: UploadFile = File(...), detections_limit: int = 20):
    """
    Submit a picture to be analyzed by the R-FCN model and get then download the picture
    with all the objects identified in the scene highlighted.

    - **detections_limit** [int]: Define the limit of objects to be ploted in the returning image.
    """
    try:
        image_file = await file.read()
        processed_image = objectdetection.get_predicted_image(image_file, TENSORFLOW_URL, detections_limit)
        filename = file.filename.split(".")[0]
        extension = file.filename.split(".")[-1]
        filename = "".join([filename, " (processed).", extension])

        return StreamingResponse(io.BytesIO(processed_image),
                                 headers={'Content-Disposition': 'attachment; filename=' + filename},
                                 media_type="image/jpg")

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
