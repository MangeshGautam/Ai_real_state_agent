from fastapi import FastAPI
from app.routes.routes import router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # ✅ Allow all origins
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "Hello FastAPI 🚀"}


@app.get("/users/{user_id}")
def get_user(user_id: int):
    return {"user_id": user_id}

app.include_router(router, prefix="/api")

