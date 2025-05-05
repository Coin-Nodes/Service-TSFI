##################################################
##  Author:
##  Date:
##################################################

import uvicorn
from fastapi import FastAPI
from starlette.responses import RedirectResponse

from api.v1.router import api_router

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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
