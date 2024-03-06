from fastapi import status
from fastapi import FastAPI
from fastapi import File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

from inference import summarize_code, generate_caption, generate_text

app = FastAPI()

@app.get('/')
async def health_check():
    return {"Status": "Ok"}


@app.post("/text_generation")
async def text_generation(input_text: str, max_length: int):
    """Generates text based on fine-tuned dataset"""
    try:
        text = generate_text(input_text, max_length)
    except Exception:
        return {"message": "There was an error uploading the file"}
    
    response =  {"input_text": input_text, "max_length": max_length, 'generated_text': text}

    return JSONResponse(content=jsonable_encoder(response), status_code=status.HTTP_200_CREATED)


@app.post("/image_captioning")
async def image_captioning(image: UploadFile = File(...)):
    """Generates caption based on fine-tuned dataset"""
    try:
        contents = image.file.read()
        with open(image.filename, 'wb') as f:
            f.write(contents)

        caption = generate_caption(image)

    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        image.file.close()

    response = {"image caption": caption}

    return JSONResponse(content=jsonable_encoder(response), status_code=status.HTTP_200_CREATED)


@app.post("/code_summary")
async def code_summary(code_block: str):
    """Generates DocStrings based on fine-tuned dataset"""
    try:
        code = summarize_code(code_block)
    except Exception:
        return {"message": "There was an error uploading the file"}
    
    response = {"code_block": code_block, 'doc_string': code}

    return JSONResponse(content=jsonable_encoder(response), status_code=status.HTTP_200_CREATED)