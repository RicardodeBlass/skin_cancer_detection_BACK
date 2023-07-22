from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
import tensorflow as tf
import uvicorn
import os
import numpy as np
import cv2
import io
from face_rec.face_detection import annotate_face

app = FastAPI()

# # Allow all requests (optional, good for development purposes)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allows all origins
#     allow_credentials=True,
#     allow_methods=["*"],  # Allows all methods
#     allow_headers=["*"],  # Allows all headers
# )

@app.get("/")
def index():
    return {"status": "ok"}

@app.post('/upload_image')
async def receive_image(img: UploadFile=File(...)):
    ### Receiving and decoding the image
    contents = await img.read()

    #contents = contents.resize((128, 128))

    #contents=np.array(contents)

    nparr = np.fromstring(contents, np.uint8)
    cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # type(cv2_img) => numpy.ndarray
    print("---------------------------------------------------------------------------------")
    print(cv2_img.shape)
    print(type(cv2_img))
    cv2_img = np.resize(cv2_img,(1,128,128,3))
    print("---------------------------------------------------------------------------------")
    print(cv2_img.shape)
    image = cv2_img/255.

    ### Do cool stuff with your image.... For example face detection
    print(os.getcwd())
    model = tf.keras.models.load_model('fast_api/model_data_SMOTE/model_data_SMOTE')
    result = model.predict(image)
    print(result)
    ### Encoding and responding with the image
    resultado=''
    if result[0][0]==result.max():
        resultado = 'Melanoma'
    elif result[0][1]==result.max():
        resultado = 'Melanocytic nevi'
    elif result[0][2]==result.max():
        resultado = 'Basal cell carcinoma'
    elif result[0][3]==result.max():
        resultado = "Actinic keratoses and intraepithelial carcinoma / Bowen's disease "
    elif result[0][4]==result.max():
        resultado = 'Benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses'
    elif result[0][5]==result.max():
        resultado = 'Dermatofibroma'
    else:
        resultado = 'Vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage'
    return JSONResponse(content=resultado)
