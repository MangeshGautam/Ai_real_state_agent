from fastapi import FastAPI
from app.routes.routes import router

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello FastAPI 🚀"}


@app.get("/users/{user_id}")
def get_user(user_id: int):
    return {"user_id": user_id}

app.include_router(router, prefix="/api")

