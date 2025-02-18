from fastapi import APIRouter, File, UploadFile
from fastapi.responses import Response
from PIL import Image, ImageDraw
import io
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

router = APIRouter()

# ✅ Set TensorFlow to use minimal memory (helps on Render)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU Memory Error: {e}")

# ✅ Lazy model loading (prevents memory overload)
def load_model():
    print("Loading TensorFlow model...")
    return hub.load("https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2")

# ✅ Process an uploaded image
@router.post("/process-image")
async def process_image(file: UploadFile = File(...)):
    try:
        # ✅ Read and open the image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # ✅ Convert image to Tensor (Model expects batch dimension)
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0).astype(np.float32)

        # ✅ Load the model only when needed
        model = load_model()

        # ✅ Apply model transformation
        stylized_image = model(tf.constant(image))[0]

        # ✅ Convert back to PIL Image
        stylized_image = np.array(stylized_image[0]) * 255
        stylized_image = Image.fromarray(np.uint8(stylized_image))

        # ✅ Save output to bytes
        output_buffer = io.BytesIO()
        stylized_image.save(output_buffer, format="PNG")
        output_buffer.seek(0)

        # ✅ Return the processed image as a response
        return Response(content=output_buffer.getvalue(), media_type="image/png")

    except Exception as e:
        return {"error": f"Failed to process image: {str(e)}"}