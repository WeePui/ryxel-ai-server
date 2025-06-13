import tempfile
import time
from flask import Flask, request, jsonify
import pickle
import cv2
import numpy as np
import math
import onnxruntime
from onnxruntime.capi import _pybind_state as C
import os
import queue
import threading
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from underthesea import word_tokenize
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Sample data sản phẩm (bạn có thể thay thế hoặc load từ cơ sở dữ liệu)
PRODUCTS_PATH = Path(__file__).parent / "products.json"
with open(PRODUCTS_PATH, "r", encoding="utf-8") as f:
    products = json.load(f)
CART_PKL_PATH = Path(__file__).parent / "cart_rules.pkl"
MODEL_PATH = Path(__file__).parent / "best.onnx"

# Hàm tiền xử lý: kết hợp các thuộc tính của sản phẩm thành 1 chuỗi mô tả
def combine_product_text(product):
    # Ghép nối các trường cơ bản của sản phẩm
    text = f"{product.get('name', '')} {product.get('description', '')} {product.get('brand', '')}"
    # Thêm thông tin của từng variant: tên và các specifications
    for variant in product.get("variants", []):
        specs = " ".join([f"{k}: {v}" for k, v in variant.get("specifications", {}).items()])
        text += f" {variant.get('name', '')} {specs}"
    return text

# Hàm tokenize cho tiếng Việt sử dụng underthesea
def viet_tokenizer(text):
    return word_tokenize(text)

# Lớp xây dựng mô hình gợi ý dựa trên nội dung sản phẩm
class ContentBasedRecommender:
    def __init__(self, products):
        self.products = products
        self.product_texts = [combine_product_text(product) for product in products]
        # Sử dụng TF-IDF với hàm tokenizer cho tiếng Việt, token_pattern=None cần thiết khi dùng tokenizer riêng
        self.vectorizer = TfidfVectorizer(tokenizer=viet_tokenizer, token_pattern=None)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.product_texts)
        # Tính độ tương đồng cosine giữa các sản phẩm
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)
    
    def recommend_by_index(self, index, top_n):
        similarity_scores = list(enumerate(self.similarity_matrix[index]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        # Bỏ đi sản phẩm gốc (score = 1) và lấy top_n sản phẩm tương tự nhất
        similar_scores = [(i, score) for i, score in similarity_scores if i != index][:top_n]
        return similar_scores

    def recommend_by_id(self, product_id, top_n):
        # Tìm chỉ số của sản phẩm theo _id
        index = next((i for i, product in enumerate(self.products) if product['_id'] == product_id), None)
        if index is None:
            return []
        similar_scores = self.recommend_by_index(index, top_n)
        # Chỉ trả về danh sách ID của các sản phẩm gợi ý và điểm tương tự
        recommended =[{
            "productID": self.products[i]['_id'], "confidence": score}
            for i, score in similar_scores]
        return recommended


# Khởi tạo mô hình gợi ý từ dữ liệu sản phẩm
recommender = ContentBasedRecommender(products)

app = Flask(__name__)

# These label lists and your helper functions remain unchanged.
__labels = [
    "FEMALE_GENITALIA_COVERED",
    "FACE_FEMALE",
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "ANUS_EXPOSED",
    "FEET_EXPOSED",
    "BELLY_COVERED",
    "FEET_COVERED",
    "ARMPITS_COVERED",
    "ARMPITS_EXPOSED",
    "FACE_MALE",
    "BELLY_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_COVERED",
    "FEMALE_BREAST_COVERED",
    "BUTTOCKS_COVERED",
]

target_tag = [
    "FEMALE_GENITALIA_COVERED",
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "ANUS_EXPOSED",
    "FEET_EXPOSED",
    "ARMPITS_EXPOSED",
    "BELLY_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_COVERED",
    "FEMALE_BREAST_COVERED",
    "BUTTOCKS_COVERED",
]


main_server_url = os.getenv("MAIN_SERVER_URL", "http://localhost:8000/api/v1/api-callback/nsfw_detected")
onnx_session = onnxruntime.InferenceSession(
    MODEL_PATH,
    providers=onnxruntime.get_available_providers()
)


def _read_image(image_path, target_size=320):
    img = cv2.imread(image_path)
    img_height, img_width = img.shape[:2]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    aspect = img_width / img_height

    if img_height > img_width:
        new_height = target_size
        new_width = int(round(target_size * aspect))
    else:
        new_width = target_size
        new_height = int(round(target_size / aspect))

    resize_factor = math.sqrt(
        (img_width ** 2 + img_height ** 2) / (new_width ** 2 + new_height ** 2)
    )
    img = cv2.resize(img, (new_width, new_height))
    pad_x = target_size - new_width
    pad_y = target_size - new_height
    pad_top, pad_bottom = [int(i) for i in np.floor([pad_y, pad_y]) / 2]
    pad_left, pad_right = [int(i) for i in np.floor([pad_x, pad_x]) / 2]

    img = cv2.copyMakeBorder(
        img,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0],
    )
    img = cv2.resize(img, (target_size, target_size))
    image_data = img.astype("float32") / 255.0  # normalize
    image_data = np.transpose(image_data, (2, 0, 1))
    image_data = np.expand_dims(image_data, axis=0)
    return image_data, resize_factor, pad_left, pad_top


def _postprocess(output, resize_factor, pad_left, pad_top):
    outputs = np.transpose(np.squeeze(output[0]))
    rows = outputs.shape[0]
    scores = []
    class_ids = []
    for i in range(rows):
        classes_scores = outputs[i][4:]
        max_score = np.amax(classes_scores)
        if max_score >= 0.2:  # Confidence threshold
            class_id = np.argmax(classes_scores)
            class_ids.append(class_id)
            scores.append(max_score)

    class_scores = {}
    for class_id, score in zip(class_ids, scores):
        class_name = __labels[class_id]
        if class_name in target_tag:
            if class_name not in class_scores or score > class_scores[class_name]:
                class_scores[class_name] = float(score)
    detections = [{"class": class_name, "score": score}
                  for class_name, score in class_scores.items()]
    return detections


def detect_nsfw(image_path):
    # Use the global 'onnx_session' instead of creating a new one.
    model_inputs = onnx_session.get_inputs()
    input_shape = model_inputs[0].shape
    input_width = input_shape[2]  # e.g. 320
    input_height = input_shape[3]  # e.g. 320
    input_name = model_inputs[0].name

    preprocessed_image, resize_factor, pad_left, pad_top = _read_image(image_path, input_width)
    outputs = onnx_session.run(None, {input_name: preprocessed_image})
    detections = _postprocess(outputs, resize_factor, pad_left, pad_top)
    return detections

# ------------------------------------------------------------------------------
# New /detect endpoint that processes a JSON payload with reviewId and image URLs.
# ------------------------------------------------------------------------------
request_queue = queue.Queue()

# URL of the main server endpoint to notify the detection result


def nsfw_worker():
    """Background thread function to process the queue."""
    while True:
        try:
            # Get the next request from the queue
            data = request_queue.get()
            if data is None:
                break  # Exit the worker thread if `None` is received

            review_id = data.get('reviewId')
            image_urls = data.get('images')

            is_valid = True  # Assume the review is valid until an invalid image is found
            all_detections = []

            # Define truly NSFW categories that should trigger rejection
            nsfw_categories = [
                "BUTTOCKS_EXPOSED",
                "FEMALE_BREAST_EXPOSED", 
                "FEMALE_GENITALIA_EXPOSED",
                "MALE_GENITALIA_EXPOSED",
                "ANUS_EXPOSED"
            ]

            for image_url in image_urls:
                # Download the image from the Cloudinary link
                response = requests.get(image_url, stream=True)
                if response.status_code != 200:
                    # If an image cannot be downloaded, skip it but don't mark as NSFW
                    print(f"Failed to download image: {image_url}")
                    continue

                # Save the downloaded image to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                    tmp_file.write(response.content)
                    temp_image_path = tmp_file.name

                try:
                    # Run NSFW detection on this image
                    detections = detect_nsfw(temp_image_path)
                    
                    # Check if any detections are actually NSFW with high confidence
                    for detection in detections:
                        class_name = detection.get("class")
                        score = detection.get("score", 0)
                        
                        # Only flag as NSFW if it's a truly inappropriate category with high confidence
                        if class_name in nsfw_categories and score >= 0.6:
                            is_valid = False
                            all_detections.append(detection)
                            break
                    
                    # If this image was flagged as NSFW, stop processing other images
                    if not is_valid:
                        break
                        
                except Exception as e:
                    print(f"Error processing image {image_url}: {str(e)}")
                    continue
                finally:
                    # Clean up the temporary file
                    if os.path.exists(temp_image_path):
                        os.remove(temp_image_path)

            # Notify the main server
            post_payload = {
                "reviewId": review_id,
                "detected": all_detections,
                "isValid": is_valid
            }
            print(f"NSFW Detection Result: {post_payload}")
            requests.post(main_server_url, json=post_payload)

        except Exception as e:
            print(f"Error in NSFW worker: {str(e)}")
            # Notify the main server in case of an error, but don't delete the review
            error_payload = {
                "reviewId": data.get('reviewId', 'unknown'),
                "isValid": True,  # Default to valid if there's an error
                "error": str(e)
            }
            requests.post(main_server_url, json=error_payload)

        finally:
            # Mark the task as done
            request_queue.task_done()

# Start the worker thread
worker_thread = threading.Thread(target=nsfw_worker, daemon=True)
worker_thread.start()

@app.route('/api/detect', methods=['POST'])
def nsfw_detection_endpoint():
    """Endpoint to enqueue NSFW detection requests."""
    data = request.get_json()
    if not data or 'reviewId' not in data or 'images' not in data:
        return jsonify({
            "status": "error",
            "message": "reviewId and images list are required."
        }), 400

    # Add the request to the queue
    request_queue.put(data)
    return jsonify({"status": "success", "message": "Request queued for processing."}), 202

@app.route('/shutdown', methods=['POST'])
def shutdown_worker():
    """Endpoint to gracefully stop the worker thread."""
    request_queue.put(None)  # Enqueue a sentinel value to stop the worker
    worker_thread.join()  # Wait for the worker thread to finish
    return jsonify({"status": "success", "message": "Worker thread stopped."}), 200


@app.route('/api/recommend', methods=['GET'])
def recommend_product():
    # Load the saved rules
    with open(CART_PKL_PATH, "rb") as f:
        loaded_rules = pickle.load(f)

    # Parse the JSON body of the request
    product_id = request.args.get('product_id')
    if not product_id:
        return jsonify({"error": "Missing product_id parameter"}), 400
    
    # Filter out rules that contain the same productID or variantID in both antecedents and consequents
    filtered_rules = loaded_rules[
        ~loaded_rules["antecedents"].apply(lambda x: product_id in x) &
        ~loaded_rules["consequents"].apply(
            lambda x: product_id in x)
    ]

    # Sort by confidence in descending order
    relevant_rules = filtered_rules.sort_values(
        by="confidence", ascending=False)

    # Create an empty list to store unique recommendations
    unique_recommendations = []
    seen_product_ids = set()

    # Iterate through the sorted rules to collect unique productIDs
    for _, row in relevant_rules.iterrows():
        # Extract antecedents (productIDs)
        product_ids = list(row["antecedents"])
        if len(product_ids) == 1:  # Ensure antecedents are a single productID
            product_id = product_ids[0]
            if product_id not in seen_product_ids:
                seen_product_ids.add(product_id)
                unique_recommendations.append({
                    "productID": product_id,
                    "confidence": row["confidence"]
                })
                # Stop when we've collected 10 unique recommendations
                if len(unique_recommendations) == 5:
                    break

    # Return recommendations in the response
    return jsonify({
        "status": "success",
        "recommendations": unique_recommendations
    }), 200
    
@app.route('/api/similar', methods=['GET'])
def recommend():
    product_id = request.args.get('product_id')
    if not product_id:
        return jsonify({"error": "Missing product_id parameter"}), 400
    recommendations = recommender.recommend_by_id(product_id, 5)
    return jsonify({
        "status": "success",
        "similars": recommendations
    }), 200

@app.route('/ping', methods=['GET'])
def ping():
    """Health check endpoint"""
    return jsonify({
        "status": "success",
        "message": "Ryxel AI Server is running",
        "timestamp": time.time(),
        "uptime": "OK"
    }), 200

@app.route('/api/health', methods=['GET'])
def health_check():
    """Detailed health check endpoint"""
    try:
        # Check if essential components are loaded
        model_loaded = onnx_session is not None
        products_loaded = len(products) > 0
        recommender_loaded = recommender is not None
        
        return jsonify({
            "status": "healthy",
            "message": "All systems operational",
            "timestamp": time.time(),
            "components": {
                "onnx_model": model_loaded,
                "products_data": products_loaded,
                "recommender": recommender_loaded,
                "total_products": len(products)
            }
        }), 200
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "message": f"Health check failed: {str(e)}",
            "timestamp": time.time()
        }), 500

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
