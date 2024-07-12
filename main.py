# main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from feature_extraction import get_feature_extractor
from similarity_search import build_similarity_model, find_similar_images
from mongo_utils import get_product_data, get_category_mapping
import tensorflow as tf
from PIL import Image
import io
import numpy as np
import os

# Load the fine-tuned model
model_path = 'model_finetuned.keras'
model = tf.keras.models.load_model(model_path)
feature_extractor = get_feature_extractor(model_path)

# Build the similarity model using the recon folder
product_data = get_product_data()
category_mapping = get_category_mapping()
recon_folder = 'product_images/recon'
nn_models, image_paths = build_similarity_model(product_data, feature_extractor, recon_folder)

app = FastAPI()

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    image = image.resize((128, 128))
    image_np = np.array(image) / 255.0
    image_np = np.expand_dims(image_np, axis=0)

    try:
        top_categories, similar_images = find_similar_images(image_np, feature_extractor, model, nn_models, image_paths, model.class_indices)
        similar_image_ids = [os.path.basename(img).split('_')[0] for img in similar_images]
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

    return JSONResponse(content={
        "Top Predicted Categories": top_categories,
        "Similar Image IDs": similar_image_ids
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
