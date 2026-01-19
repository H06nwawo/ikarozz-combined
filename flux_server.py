#!/usr/bin/env python3
import os
import io
import base64
import torch
from flask import Flask, request, jsonify
from diffusers import FluxPipeline
from PIL import Image

app = Flask(__name__)

pipe = None

def load_model():
    global pipe
    print("Loading Flux Schnell model...")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.bfloat16
    )
    pipe.to("cuda")
    pipe.enable_model_cpu_offload()
    print("Model loaded successfully!")

@app.route('/v1/images/generations', methods=['POST'])
def generate_image():
    try:
        data = request.get_json()
        
        prompt = data.get('prompt', '')
        if not prompt:
            return jsonify({'error': 'Prompt is required'}), 400
        
        num_inference_steps = data.get('num_inference_steps', 4)
        guidance_scale = data.get('guidance_scale', 0.0)
        width = data.get('width', 1024)
        height = data.get('height', 1024)
        
        if width % 8 != 0 or height % 8 != 0:
            return jsonify({'error': 'Width and height must be multiples of 8'}), 400
        
        print(f"Generating image with prompt: {prompt}")
        
        image = pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=torch.Generator("cuda").manual_seed(0)
        ).images[0]
        
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        response = {
            "created": int(torch.cuda.Event(enable_timing=False).record().query()),
            "data": [
                {
                    "b64_json": img_str,
                    "revised_prompt": prompt
                }
            ]
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=8001, debug=False)
