from fastapi import APIRouter, FastAPI

from apps.pyserver.app.api.v1.NSTW import NSTW, oracle

api_router = APIRouter()
app = FastAPI()

api_router.include_router(NSTW.router, tags=["NSTW"])
api_router.include_router(oracle.router, tags=["oracle"])
