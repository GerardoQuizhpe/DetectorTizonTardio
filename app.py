from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import cv2
from ultralytics import YOLO

# Configuración
class Config:
    UPLOAD_FOLDER = 'static/imagenes'
    DETECTION_FOLDER = 'static/detecciones'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    CONFIDENCE_THRESHOLD = 0.7
    MODEL_PATH = 'modelo/yolov8.pt'

# Inicialización
app = Flask(__name__)
app.config.from_object(Config)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DETECTION_FOLDER'], exist_ok=True)

model = YOLO(app.config['MODEL_PATH'])
model.conf = app.config['CONFIDENCE_THRESHOLD']

# Funciones auxiliares
def is_allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_image(filepath):
    # Definir tamaños fijos para la imagen de salida
    TARGET_WIDTH = 640
    TARGET_HEIGHT = 480
    
    # Constantes para visualización
    FIXED_LINE_THICKNESS = 2
    FIXED_FONT_SCALE = 0.7
    FIXED_PADDING = 10
    FIXED_BG_PADDING = 5
    TOP_MARGIN = 30  # Margen desde la parte superior
    
    # Leer la imagen original
    img = cv2.imread(filepath)
    if img is None:
        raise ValueError("No se pudo cargar la imagen")
    
    # Obtener dimensiones originales
    original_height, original_width = img.shape[:2]
    
    # Calcular factores de escala
    width_scale = TARGET_WIDTH / original_width
    height_scale = TARGET_HEIGHT / original_height
    
    # Redimensionar la imagen al tamaño objetivo
    resized_img = cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT))
    
    # Realizar la predicción en la imagen original
    results = model.predict(filepath)
    detections = results[0].boxes
    
    # Trabajar con la imagen redimensionada
    output_img = resized_img.copy()
    
    bboxs = []
    for box in detections:
        confidence = float(box.conf[0])
        if confidence >= app.config['CONFIDENCE_THRESHOLD']:
            # Obtener coordenadas originales
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            
            # Escalar coordenadas al nuevo tamaño
            x1_scaled = int(x1 * width_scale)
            y1_scaled = int(y1 * height_scale)
            x2_scaled = int(x2 * width_scale)
            y2_scaled = int(y2 * height_scale)
            
            bboxs.append({
                'x1': x1_scaled,
                'y1': y1_scaled,
                'x2': x2_scaled,
                'y2': y2_scaled,
                'label': f"{confidence:.2f}"
            })
    
    if bboxs:
        for box in bboxs:
            x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
            confidence = float(box['label'])
            
            # Dibujar rectángulo con grosor fijo
            cv2.rectangle(output_img, (x1, y1), (x2, y2), (255, 0, 0), FIXED_LINE_THICKNESS)
            
            # Preparar texto
            label_text = f"{confidence:.2f}"
            
            # Obtener dimensiones del texto
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                FIXED_FONT_SCALE,
                FIXED_LINE_THICKNESS
            )
            
            # Calcular posición del texto
            text_x = x1
            text_y = y1 - FIXED_PADDING if y1 - FIXED_PADDING > text_height else y1 + text_height + FIXED_PADDING
            
            # Asegurar que el texto no se salga de la imagen
            text_x = min(max(0, text_x), TARGET_WIDTH - text_width)
            text_y = min(max(text_height, text_y), TARGET_HEIGHT - baseline)
            
            # Dibujar fondo para el texto
            cv2.rectangle(
                output_img,
                (text_x - FIXED_BG_PADDING, text_y - text_height - FIXED_BG_PADDING - baseline),
                (text_x + text_width + FIXED_BG_PADDING, text_y + FIXED_BG_PADDING),
                (255, 0, 0),
                -1
            )
            
            # Dibujar texto
            cv2.putText(
                output_img,
                label_text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                FIXED_FONT_SCALE,
                (255, 255, 255),
                FIXED_LINE_THICKNESS
            )
    else:
        # Mensaje cuando no hay detecciones
        text = "No hay presencia de la enfermedad"
        
        (text_width, text_height), _ = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            FIXED_FONT_SCALE,
            FIXED_LINE_THICKNESS
        )
        
        x = (TARGET_WIDTH - text_width) // 2
        y = TOP_MARGIN  # Usar el margen superior definido
        
        cv2.putText(
            output_img,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            FIXED_FONT_SCALE,
            (255, 255, 255),
            FIXED_LINE_THICKNESS
        )
    
    # Guardar la imagen procesada
    detection_path = os.path.join(app.config['DETECTION_FOLDER'], 
                                os.path.basename(filepath))
    cv2.imwrite(detection_path, output_img, [cv2.IMWRITE_JPEG_QUALITY, 100])
    
    return detection_path, len(bboxs)

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/deteccion', methods=['POST'])
def deteccion():
    files = request.files.getlist('images')
    if len(files) > 10:
        return render_template('index.html', error_message="Solo se permiten 10 imágenes.")

    detected_images = []
    no_disease_images = []

    for file in files:
        if file and is_allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            detection_path, detection_count = process_image(filepath)
            if detection_count > 0:
                detected_images.append(os.path.basename(detection_path))
            else:
                no_disease_images.append(filename)

    return render_template(
        'deteccion.html',
        detected_images=detected_images,
        no_disease_images=no_disease_images
    )


if __name__ == '__main__':
    app.run(debug=True)
