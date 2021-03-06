from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO

import cv2
import numpy as np
from fastai.vision import *

from Inferencers.CarcinomaClassifier import Inference, CancerImageList, open_im

model_file_url = 'https://www.dropbox.com/s/emgt2k567hmlo25/breast_cancer_1024-98acc.pkl?dl=1'
model_file_name = 'breast_cancer_1024-98acc.pkl'
classes = ['Normal', 'Benign', 'Invasive', 'InSitu']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_learner():
    # await download_file(model_file_url, path/'models'/model_file_name)
    try:
        predictor = Inference(path/'models', model_file_name)
        return predictor
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else: raise
    return predictor


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
predictor = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route('/')
def index(request):
    html = path/'view'/'index.html'
    return HTMLResponse(html.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    data = await request.form()
    img_bytes = await (data['file'].read())
    img = open_im(BytesIO(img_bytes))
    return JSONResponse({'result': str(predictor(img)[0])})
    
if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app, host='0.0.0.0', port=8080)

