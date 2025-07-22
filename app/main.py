from fastapi import FastAPI
from app.core_router import core_router
from app.v1_router import v1_router
from app.v2_router import v2_router

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


app.include_router(core_router)
app.include_router(v1_router)
app.include_router(v2_router)
