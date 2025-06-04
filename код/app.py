import os


os.environ['OPENBLAS_NUM_THREADS'] = '1'


from fastapi import FastAPI, Request, UploadFile, HTTPException, status, File, Body, Response
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from typing import Annotated
import pandas as pd
import aiofiles
import datetime
import uvicorn
import sqlite3
import io
import tempfile
import uuid



from misc.predictor import get_prediction


# Создание FastAPI приложения
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


dummy_storage = {} # временное хранилище данных

# тестовая функция для проверки работы ML-модели
def get_response_df(input_df: pd.DataFrame) -> pd.DataFrame:

    X = pd.DataFrame({"data": [i for i in range(10)]})
    Y_ = get_prediction(X)

    inference_df = X
    inference_df["predict"] = Y_
    return inference_df

# API Endpoints
@app.post("/predict/")
async def predict_pep(request: Request, file: UploadFile = File(None)):
    try:
        data = await request.json()
    except:
        data = None

    if data:
        # Обработка json файла
        input_df = pd.DataFrame(data) # JSON → DataFrame
        result = get_prediction(input_df)
    elif file:
        # Обработка ввода файла
        content = await file.read() # Чтение файла
        input_df = pd.read_json(io.BytesIO(content)) # Парсинг JSON-файла
        result = get_prediction(input_df) # Предсказание ML-модели
    else:
        return {"error": "No valid input provided"}
    inference_df = pd.DataFrame({"id": [i for i in range(len(result))], "result": result})
    return inference_df.to_dict(orient="records")

# Вывод на HTML-страницу результата предсказания
@app.get("/predict/{unique_id}/", response_class=HTMLResponse)
async def predict_gep(request: Request, unique_id: str):

    json_data_str = dummy_storage.get(unique_id)
    
    if json_data_str is None:
        raise HTTPException(status_code=404, detail="Processed data not found")
    
    # Optionally, you can perform some operations on the DataFrame here
    json_data = eval(json_data_str)
    inference_df = pd.DataFrame(json_data)
    result = get_prediction(inference_df)
    columns = ['id', 'result']
    data = [[i, result[i]] for i in range(len(result))]
    
    # Return the processed data
    return templates.TemplateResponse(
        'table.html',
        {'request': request, 'columns': columns, 'data': data, 'for_download': str(pd.DataFrame({"id": [i for i in range(len(result))], "result": result}).to_json(orient="records"))}
    )


# Temporary storage for processed data
temp_storage = {}


@app.post("/upload/")
async def upload(file: UploadFile = File(None)):
    global dummy_storage
    if file:
        # Обработка ввода файла
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file_path = temp_file.name
            try:
                # Запись содержимого файла во временный файл
                while content := await file.read(1024):  # считывание порциями по 1024 bytes
                    temp_file.write(content)
                temp_file.flush()
                temp_file.close()
                
                # Обработка временного файла
                df = pd.read_json(temp_file_path)
            finally:
                # Очистка
                os.remove(temp_file_path)
    else:
        raise HTTPException(status_code=400, detail="No valid input provided")

    # Преобразование данных обратно в json для получения ответа
    json_data = df.to_dict(orient="records")
    
    # Генерация уникального идентификатора
    unique_id = str(uuid.uuid4())

    dummy_storage[unique_id] = str(json_data) # Сохранение данных
    
    # Перенаправление на другую конечную точку с уникальным идентификатором
    return RedirectResponse(url=f"/predict/{unique_id}/", status_code=302)

# Возвращение HTML-страницы (главную) из статического файла
@app.get('/', response_class=HTMLResponse)
async def main():
    with open("static/index.html", encoding='utf-8') as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

# Запуск
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)