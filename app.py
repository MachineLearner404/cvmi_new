from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os
import base64
from io import BytesIO
import sys
from pathlib import Path
import torch
import flask  # Import flask to get version

app = Flask(__name__)
app.secret_key = 'cvmi-secret-key-2024'  # Changed for security

USERS = {
    "daksh": {"password": "cvmi123", "name": "Daksh Vyawhare", "role": "Deep Learning Engineer"},
    "khushi": {"password": "research456", "name": "Dr. Khushi Rathod", "role": "Researcher"},
    "alka": {"password": "mentor1", "name": "Dr. Alka Banker", "role": "Mentor"},
    "rahul": {"password": "mentor2", "name": "Dr. Rahul Kumar", "role": "Mentor"},
    "demo": {"password": "demo123", "name": "Demo User", "role": "Researcher"}  # Added demo user
}

class_color_map = {
    "C2": [139, 0, 0],    # Dark Red
    "C3": [0, 100, 0],    # Dark Green  
    "C4": [0, 0, 139],    # Dark Blue
}

# CVMI Stage Information
CVMI_STAGE_INFO = {
    "CS1": {
        "name": "Stage 1 - Initiation",
        "description": "Lower borders of C2, C3, C4 are flat. Vertebral bodies are trapezoid in shape.",
        "age_range": "6-8 years",
        "characteristics": ["Flat lower borders", "Trapezoid shapes", "No concavities"]
    },
    "CS2": {
        "name": "Stage 2 - Acceleration", 
        "description": "Concavities in lower borders of C2 and C3. C4 still flat.",
        "age_range": "8-10 years",
        "characteristics": ["C2 & C3 concavities", "C4 flat", "Beginning maturation"]
    },
    "CS3": {
        "name": "Stage 3 - Transition",
        "description": "Concavities in lower borders of C2, C3, C4. C3 and C4 are rectangular.",
        "age_range": "10-12 years", 
        "characteristics": ["All concavities present", "C3 & C4 rectangular", "Active growth"]
    },
    "CS4": {
        "name": "Stage 4 - Deceleration",
        "description": "Distinct concavities in all vertebrae. C3 and C4 nearly square.",
        "age_range": "12-14 years",
        "characteristics": ["Deep concavities", "Square shapes", "Growth slowing"]
    },
    "CS5": {
        "name": "Stage 5 - Maturation",
        "description": "More accentuated concavities. All vertebrae are rectangular.",
        "age_range": "14-16 years", 
        "characteristics": ["Accentuated concavities", "All rectangular", "Near completion"]
    },
    "CS6": {
        "name": "Stage 6 - Completion",
        "description": "Deep concavities, fully developed vertebrae. Growth complete.",
        "age_range": "16-18 years",
        "characteristics": ["Deep concavities", "Fully developed", "Growth complete"]
    }
}

# Global variable to store analysis data
analysis_data = {}

# Cross-platform model loading
def load_models():
    """
    Load YOLO models with cross-platform compatibility
    Handles Windows, Mac (Intel & Apple Silicon), and Linux
    """
    try:
        print("=" * 60)
        print("INITIALIZING MODELS...")
        print("=" * 60)
        
        # Get the directory where your script is located
        base_dir = Path(__file__).parent
        model_dir = base_dir / "model"
        
        # Define model paths
        roi_path = model_dir / "roi_best.pt"
        seg_path = model_dir / "seg_best.pt"
        cls_path = model_dir / "best.pt"
        
        # Check if model directory exists
        if not model_dir.exists():
            print(f"❌ Model directory not found: {model_dir}")
            print("Please create a 'model' folder with the required .pt files")
            return None, None, None
        
        # Check if model files exist
        model_files = {
            "ROI": roi_path,
            "Segmentation": seg_path,
            "Classification": cls_path
        }
        
        missing_models = []
        for name, path in model_files.items():
            if path.exists():
                print(f"✅ {name} model found: {path}")
            else:
                print(f"❌ {name} model not found: {path}")
                missing_models.append(name)
        
        if missing_models:
            print(f"Missing models: {', '.join(missing_models)}")
            return None, None, None
        
        # Detect available device
        device = detect_device()
        print(f"🖥️  Using device: {device}")
        
        # Load models with error handling
        models = {}
        for name, path in model_files.items():
            try:
                print(f"📥 Loading {name} model...")
                model = YOLO(str(path))
                model.to(device)
                models[name.lower()] = model
                print(f"✅ {name} model loaded successfully!")
            except Exception as e:
                print(f"❌ Failed to load {name} model: {e}")
                return None, None, None
        
        # Verify models are working
        if verify_models(*models.values()):
            print("🎉 All models loaded and verified successfully!")
            return models['roi'], models['segmentation'], models['classification']
        else:
            print("❌ Model verification failed!")
            return None, None, None
        
    except Exception as e:
        print(f"❌ Unexpected error during model loading: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def detect_device():
    """Detect and return the best available device"""
    try:
        if torch.backends.mps.is_available():
            return 'mps'  # Apple Silicon Macs
        elif torch.cuda.is_available():
            return 'cuda'  # NVIDIA GPUs
        else:
            return 'cpu'   # CPU fallback
    except:
        return 'cpu'  # Fallback to CPU if any error

def verify_models(roi_model, seg_model, cls_model):
    """Verify that all models are loaded and functional"""
    try:
        print("🔍 Verifying models...")
        
        # Create a small test image
        test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        # Test ROI model
        roi_result = roi_model(test_image, verbose=False)
        print("✅ ROI model verified")
        
        # Test segmentation model
        seg_result = seg_model(test_image, verbose=False)
        print("✅ Segmentation model verified")
        
        # Test classification model
        cls_result = cls_model(test_image, verbose=False)
        print("✅ Classification model verified")
        
        return True
        
    except Exception as e:
        print(f"❌ Model verification failed: {e}")
        return False

# Initialize models
print("🚀 Starting CVMI Analysis System...")
roi_model, seg_model, cls_model = load_models()

def numpy_to_base64(image_array):
    """Convert numpy array to base64 string with error handling"""
    try:
        if image_array.dtype != np.uint8:
            image_array = image_array.astype(np.uint8)
        
        # Handle different image formats
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            pil_img = Image.fromarray(image_array)
        elif len(image_array.shape) == 2:
            pil_img = Image.fromarray(image_array, mode='L')
        else:
            # Convert to RGB if in BGR format
            pil_img = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
        
        buff = BytesIO()
        pil_img.save(buff, format="JPEG", quality=95)
        return base64.b64encode(buff.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        # Return a placeholder image
        placeholder = np.ones((100, 100, 3), dtype=np.uint8) * 255
        cv2.putText(placeholder, "Error", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        return numpy_to_base64(placeholder)

def create_cvmi_chart(cvmi_stage):
    """Create a CVMI chart image with the predicted stage highlighted"""
    try:
        # Create a blank chart image
        chart_width, chart_height = 800, 500
        chart = np.ones((chart_height, chart_width, 3), dtype=np.uint8) * 255
        
        # Draw chart title
        cv2.putText(chart, "CERVICAL VERTEBRAL MATURATION INDEX (CVMI)", (150, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        
        # Draw stages
        stages = list(CVMI_STAGE_INFO.keys())
        stage_width = chart_width // len(stages)
        
        for i, stage in enumerate(stages):
            x_center = i * stage_width + stage_width // 2
            y_center = 200
            
            # Draw stage circle
            color = (150, 150, 150)  # Default gray
            text_color = (0, 0, 0)
            if stage == cvmi_stage:
                color = (0, 150, 255)  # Orange for predicted stage
                text_color = (0, 0, 255)
            
            cv2.circle(chart, (x_center, y_center), 50, color, -1)
            cv2.circle(chart, (x_center, y_center), 50, (0, 0, 0), 2)
            
            # Stage label
            cv2.putText(chart, stage, (x_center - 20, y_center + 8), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Stage name
            stage_name = CVMI_STAGE_INFO[stage]["name"].split(' - ')[1]
            cv2.putText(chart, stage_name, (x_center - 45, y_center + 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
            
            # Age range
            cv2.putText(chart, CVMI_STAGE_INFO[stage]["age_range"], (x_center - 40, y_center + 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
        
        # Highlight predicted stage with arrow
        if cvmi_stage in stages:
            predicted_index = stages.index(cvmi_stage)
            x_center = predicted_index * stage_width + stage_width // 2
            
            # Draw arrow pointing to predicted stage
            arrow_start = (x_center, 300)
            arrow_end = (x_center, 260)
            cv2.arrowedLine(chart, arrow_start, arrow_end, (0, 0, 255), 3, tipLength=0.3)
            
            cv2.putText(chart, f"PREDICTED STAGE: {cvmi_stage}", (x_center - 100, 350), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return chart
    except Exception as e:
        print(f"Error creating CVMI chart: {e}")
        # Return a simple error chart
        error_chart = np.ones((500, 800, 3), dtype=np.uint8) * 255
        cv2.putText(error_chart, "Error creating chart", (200, 250), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return error_chart

def create_comparison_view(segmented_image_b64, predicted_stage):
    """Create a comparison view showing segmented image and corresponding CS stage image"""
    try:
        # Decode segmented image
        segmented_bytes = base64.b64decode(segmented_image_b64)
        segmented_img = np.array(Image.open(BytesIO(segmented_bytes)))
        
        # Try to load the corresponding CS stage image
        stage_number = predicted_stage.replace('CS', '')
        stage_image_path = f"static/images/stage{stage_number}.jpg"
        
        if os.path.exists(stage_image_path):
            stage_img = cv2.imread(stage_image_path)
            stage_img = cv2.cvtColor(stage_img, cv2.COLOR_BGR2RGB)
        else:
            # Create a placeholder if image doesn't exist
            stage_img = np.ones((400, 400, 3), dtype=np.uint8) * 200
            cv2.putText(stage_img, f"CS{stage_number} Reference", (50, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
            cv2.putText(stage_img, "Image not found", (80, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
        
        # Resize images to same height for comparison
        target_height = 400
        seg_ratio = target_height / segmented_img.shape[0]
        seg_width = int(segmented_img.shape[1] * seg_ratio)
        segmented_resized = cv2.resize(segmented_img, (seg_width, target_height))
        
        stage_ratio = target_height / stage_img.shape[0]
        stage_width = int(stage_img.shape[1] * stage_ratio)
        stage_resized = cv2.resize(stage_img, (stage_width, target_height))
        
        # Create comparison layout
        comparison = np.ones((target_height + 100, max(seg_width, stage_width) * 2 + 50, 3), dtype=np.uint8) * 255
        
        # Place segmented image
        seg_x = (comparison.shape[1] // 2 - seg_width - 25)
        comparison[50:50+target_height, seg_x:seg_x+seg_width] = segmented_resized
        
        # Place stage reference image
        stage_x = (comparison.shape[1] // 2 + 25)
        comparison[50:50+target_height, stage_x:stage_x+stage_width] = stage_resized
        
        # Add labels
        cv2.putText(comparison, "PATIENT'S VERTEBRAE", (seg_x, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 0), 2)
        cv2.putText(comparison, f"REFERENCE: {predicted_stage}", (stage_x, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 150), 2)
        
        # Add separator line
        mid_x = comparison.shape[1] // 2
        cv2.line(comparison, (mid_x, 50), (mid_x, 50+target_height), (200, 200, 200), 2)
        
        return numpy_to_base64(comparison)
        
    except Exception as e:
        print(f"Error creating comparison view: {e}")
        return segmented_image_b64  # Fallback to original segmented image

@app.route('/')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    return redirect(url_for('main_page'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if username in USERS and USERS[username]['password'] == password:
            session['username'] = username
            session['name'] = USERS[username]['name']
            flash(f'Welcome back, {USERS[username]["name"]}!', 'success')
            return redirect(url_for('main_page'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('login'))

@app.route('/main')
def main_page():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    # Check if models are loaded
    models_loaded = all([roi_model, seg_model, cls_model])
    return render_template('main.html', 
                         name=session['name'],
                         models_loaded=models_loaded)

@app.route('/analysis')
def analysis_page():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    # Get analysis data for current user
    user_data = analysis_data.get(session['username'])
    return render_template('analysis.html', 
                         name=session['name'], 
                         analysis_data=user_data, 
                         CVMI_STAGE_INFO=CVMI_STAGE_INFO)

@app.route('/model_status')
def model_status():
    """API endpoint to check model status"""
    status = {
        'roi_model_loaded': roi_model is not None,
        'seg_model_loaded': seg_model is not None, 
        'cls_model_loaded': cls_model is not None,
        'all_models_loaded': all([roi_model, seg_model, cls_model])
    }
    return jsonify(status)

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check file type
    allowed_extensions = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
    if '.' not in file.filename or file.filename.split('.')[-1].lower() not in allowed_extensions:
        return jsonify({'error': 'Invalid file type. Please upload an image file.'}), 400
    
    try:
        print(f"🖼️ Processing image: {file.filename}")
        
        if not all([roi_model, seg_model, cls_model]):
            return jsonify({
                'error': 'Models not loaded properly. Please check the server logs and ensure all model files are in the model folder.'
            }), 500
        
        # Read and process image
        image = Image.open(file.stream).convert('RGB')
        image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        print("🔍 Detecting ROI...")
        # ROI detection
        roi_results = roi_model(image_bgr)
        boxes = roi_results[0].boxes.xyxy.cpu().numpy()
        
        if len(boxes) == 0:
            return jsonify({'error': 'No Region of Interest (ROI) detected. Please try with a different image.'}), 400
        
        x1, y1, x2, y2 = map(int, boxes[0])
        cropped_roi = image_bgr[y1:y2, x1:x2]
        
        print("📐 Performing segmentation...")
        # Segmentation
        seg_results = seg_model.predict(source=cropped_roi, imgsz=512, save=False, verbose=False)
        masks = seg_results[0].masks
        
        if masks is None or len(masks.data) == 0:
            return jsonify({'error': 'No vertebrae detected in the ROI. Please try with a clearer image.'}), 400
        
        classes = seg_results[0].boxes.cls.cpu().numpy()
        names = seg_model.names
        
        # Create segmentation visualization
        base_img = cropped_roi.copy()
        overlay = np.zeros_like(base_img, dtype=np.uint8)
        mask_data = masks.data.cpu().numpy()
        target_shape = (base_img.shape[1], base_img.shape[0])
        
        for i, mask in enumerate(mask_data):
            class_id = int(classes[i])
            class_name = names[class_id]
            color = class_color_map.get(class_name, [50, 50, 50])
            resized_mask = cv2.resize(mask, target_shape, interpolation=cv2.INTER_NEAREST)
            binary_mask = (resized_mask > 0.5).astype(np.uint8)
            colored_mask = np.zeros_like(overlay)
            for c in range(3):
                colored_mask[:, :, c] = binary_mask * color[c]
            overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.5, 0)
        
        final_img = cv2.addWeighted(base_img, 1.0, overlay, 0.5, 0)
        final_img_rgb = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
        
        print("🎯 Classifying CVMI stage...")
        # CVMI Classification
        cvmi_results = cls_model(cropped_roi, verbose=False)
        probs = cvmi_results[0].probs
        predicted_class_name = cls_model.names[probs.top1]
        
        # Map to CS format
        cvmi_stage = predicted_class_name.upper().replace('STAGE_', 'CS')
        
        # Create CVMI chart
        cvmi_chart = create_cvmi_chart(cvmi_stage)
        
        # Create comparison view
        segmented_b64 = numpy_to_base64(final_img_rgb)
        comparison_view = create_comparison_view(segmented_b64, cvmi_stage)
        
        # Convert images to base64
        original_b64 = numpy_to_base64(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        roi_b64 = numpy_to_base64(cv2.cvtColor(cropped_roi, cv2.COLOR_BGR2RGB))
        
        confidence = float(probs.data.cpu().numpy()[probs.top1])
        
        # Store analysis data globally for this user
        analysis_data[session['username']] = {
            'segmented_image': segmented_b64,
            'cvmi_chart': numpy_to_base64(cvmi_chart),
            'comparison_view': comparison_view,
            'cvmi_stage': cvmi_stage,
            'stage_info': CVMI_STAGE_INFO.get(cvmi_stage, {}),
            'confidence': confidence
        }
        
        print(f"✅ Analysis complete! Predicted stage: {cvmi_stage} (Confidence: {confidence:.2%})")
        
        return jsonify({
            'success': True,
            'original_image': original_b64,
            'roi_image': roi_b64,
            'segmented_image': segmented_b64,
            'cvmi_chart': numpy_to_base64(cvmi_chart),
            'comparison_view': comparison_view,
            'cvmi_stage': cvmi_stage,
            'confidence': confidence,
            'message': f'Analysis complete! Predicted CVMI Stage: {cvmi_stage}'
        })
        
    except Exception as e:
        print(f"❌ Error in process_image: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

@app.route('/get_analysis_data')
def get_analysis_data():
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    user_data = analysis_data.get(session['username'])
    if user_data:
        return jsonify({'success': True, **user_data})
    else:
        return jsonify({'error': 'No analysis data found. Please upload and analyze an image first.'}), 404

@app.errorhandler(404)
def not_found_error(error):
    return render_template('error.html', error="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', error="Internal server error"), 500

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 CVMI ANALYSIS SYSTEM STARTING...")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Flask version: {flask.__version__}")  # Fixed this line
    print(f"OpenCV version: {cv2.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print("=" * 60)
    
    # Create necessary directories
    os.makedirs('static/images', exist_ok=True)
    
    # Run the application
    app.run(debug=False, host='0.0.0.0', port=5000)