from fastapi import FastAPI, File, UploadFile
import shutil
import os
import uuid
from ocr_pipeline import extract_table

app = FastAPI()

# Ensure folders exist
os.makedirs("temp", exist_ok=True)
os.makedirs("Output", exist_ok=True)


@app.get("/")
def home():
    return {"message": "Hybrid OCR API is running"}


@app.post("/process/")
async def process_image(file: UploadFile = File(...)):
    try:
        # Unique filename
        file_id = str(uuid.uuid4())
        file_path = f"temp/{file_id}_{file.filename}"

        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run OCR pipeline
        table, csv_path = extract_table(file_path, f"{file_id}.csv")

        # Convert table to pipe-separated CSV string
        csv_text = "\n".join(["|".join(row) for row in table])

        return {
            "status": "success",
            "rows": len(table),
            "csv": csv_text
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)