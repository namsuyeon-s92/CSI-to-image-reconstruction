import multiprocessing
import asyncio
import base64
import json
import os
from contextlib import asynccontextmanager

import numpy as np
import torch
import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from core.preprocessor import TemporalMeshPreprocessor
from core.inferencer import BaseInferencer
from core.postprocessor import ImagePostprocessor
from models.vae import VAE


UDP_HOST = '0.0.0.0'
INFERENCE_UDP_PORT = 8000
CSI_DATA_LENGTH = 256
CSI_VALID_SUBCARRIER_INDEX = [i for i in range(6, 32)] + [i for i in range(33, 59)]
NUM_SUBCARRIERS = len(CSI_VALID_SUBCARRIER_INDEX)

inference_interval = 50
send_interval = 0.25
csi_count = 0
csi_queue = multiprocessing.Queue(maxsize=10)

if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

window_size = 151
checkpoint_path = 'trained_models/epoch=26-val_loss=389.5090.ckpt'


async def stats_printer():
    global csi_count
    while True:
        await asyncio.sleep(1.0)
        print(f'[STATS] CSI: {csi_count} Hz')
        csi_count = 0


def extract_csi_data(decoded_data):
    # Example format from rx: "MAC",rssi,...,timestamp,...,"[%d,%d...]"
    # The timestamp is the 18th element (index 17) in the CSV-like string before the array.
    # We can split by ',' up to the '['
    
    start_index = decoded_data.rfind('[')
    end_index = decoded_data.rfind(']')
    if start_index == -1 or end_index == -1:
        return None
    
    csv_part = decoded_data[:start_index]
    elements = csv_part.split(',')
    
    # Check if elements has enough length to contain timestamp
    if len(elements) < 18:
        return None
        
    try:
        timestamp = int(elements[17])
    except ValueError:
        return None

    csi_string = decoded_data[start_index:end_index+1]
    try:
        data_array = json.loads(csi_string)
        return {
            'local_timestamp': timestamp,
            'data': data_array
        }
    except:
        return None


def inference_worker(queue, checkpoint_path, window_size, valid_subcarrier_index, device):
    preprocessor = TemporalMeshPreprocessor(window_size, valid_subcarrier_index)
    inferencer = BaseInferencer(VAE, checkpoint_path, device, window_size=window_size, num_subcarriers=len(valid_subcarrier_index))
    postprocessor = ImagePostprocessor()

    while True:
        try:
            decoded_data = queue.get()
            if decoded_data is None:
                print("Inference worker received shutdown signal.")
                break

            csi_data = extract_csi_data(decoded_data)
            if csi_data is None or len(csi_data['data']) != CSI_DATA_LENGTH:
                continue

            preprocessor.add_data(csi_data)

            if not preprocessor.is_ready():
                continue

            spectrogram = preprocessor.process()
            if spectrogram is None:
                # Buffer might be large enough but strictly not covering the 1.5s time window.
                continue

            reconstruction = inferencer.infer(spectrogram)
            image = postprocessor.process(reconstruction)

            os.makedirs('media', exist_ok=True)
            np.save('media/image.npy', image)
            
            preprocessor.consume_buffer()
            
        except Exception as e:
            print(f'Inference Error: {e}')


class InferenceUdpServerProtocol:
    def connection_made(self, transport):
        print(f'UDP server started on {UDP_HOST}:{INFERENCE_UDP_PORT}')

    def datagram_received(self, data, addr):
        global csi_count
        try:
            csi_count += 1
            decoded_data = data.decode()
            if not csi_queue.full():
                csi_queue.put_nowait(decoded_data)
        except Exception as e:
            print(f'Inference UDP Error: {e}')


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_running_loop()

    inference_proc = multiprocessing.Process(target=inference_worker, args=(csi_queue, ))
    inference_proc.start()

    asyncio.create_task(stats_printer())

    inference_transport, _ = await loop.create_datagram_endpoint(
        lambda: InferenceUdpServerProtocol(),
        local_addr=(UDP_HOST, INFERENCE_UDP_PORT)
    )

    print('UDP server startup sequence finished.')
    
    yield

    print('Closing UDP server...')
    inference_transport.close()
    csi_queue.close()
    csi_queue.join_thread() 
    inference_proc.join(timeout=5)
    if inference_proc.is_alive():
        print("Worker did not terminate, forcing termination...")
        inference_proc.terminate()


app = FastAPI(lifespan=lifespan)
app.mount('/static', StaticFiles(directory='static'), name='static')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

@app.get('/')
async def get():
    return FileResponse('templates/display.html')


def load_images():
    image = np.load('media/image.npy')
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
    result, buffer = cv2.imencode('.jpeg', image, encode_param)
    if result:
        image = base64.b64encode(buffer).decode('utf-8')
        images = json.dumps({'image': image})
        return images
    else:
        return None

@app.websocket('/ws')
async def steam_csi_image(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            images = load_images()
            if images is not None:
                await websocket.send_text(images)
            await asyncio.sleep(send_interval)
    except WebSocketDisconnect:
        print("Websocket disconnected.")
    except Exception as e:
        print(f'error: {e}')
