from fastapi import FastAPI
from process_image import router  # Import the router

app = FastAPI()

# Include the routes from process_image.py
app.include_router(router)

@app.get("/")
def home():
    return {"message": "Hello, FastAPI!"}
