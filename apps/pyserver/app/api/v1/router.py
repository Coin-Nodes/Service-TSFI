from fastapi import APIRouter, FastAPI

from api.v1 import nstw, oracle

api_router = APIRouter()
app = FastAPI()

api_router.include_router(nstw.router, tags=["nstw"])
api_router.include_router(oracle.router, tags=["oracle"])
