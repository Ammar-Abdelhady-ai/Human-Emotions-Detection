from service.api.api import main_router
from fastapi import FastAPI
import onnxruntime as rt


app = FastAPI(project_name="Emotions Detection")
app.include_router(main_router)

providers = ["CPUExecutionProvider"]
output_path = r"service\vit_keras.onnx"
m_q = rt.InferenceSession(
    output_path, providers=providers)


@app.get("/")
async def read_root():
    return {"Hello": "World"}

