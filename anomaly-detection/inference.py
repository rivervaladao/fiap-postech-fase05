import logging
import sys
import os
import dotenv
from abc import ABC, abstractmethod
import cv2
from ultralytics import YOLO
import smtplib
from email.mime.text import MIMEText

dotenv.load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(filename="logs.log", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

path_model = "/home/river/workspace/experiment/fiap/FIAP-POS-TECH/fase05/datasets/runs/detect/train/weights/best.pt"

CLASS_THRESHOLDS = {
    "knife": 0.55,
    "scissors": 0.55
}

class Alert(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def send_alert(self, message, frame=None):
        pass

class ConsoleAlert(Alert):
    def send_alert(self, message, frame=None):
        logger.debug(f"Alerta: {message}")

class EmailAlert(Alert):
    def __init__(self, recipient_email):
        super().__init__()
        self.password = os.getenv("EMAIL_PASSWORD")
        if not self.password:
            logger.error("Senha do email não encontrada. Defina a variável de ambiente EMAIL_PASSWORD.")
            raise ValueError("Senha do email não encontrada. Defina a variável de ambiente EMAIL_PASSWORD.")
        
        self.from_email = os.getenv("EMAIL_FROM")
        if not self.from_email:
            logger.error("Email de origem não encontrado. Defina a variável de ambiente EMAIL_FROM.")
            raise ValueError("Email de origem não encontrado. Defina a variável de ambiente EMAIL_FROM.")
        
        self.recipient_email = recipient_email

    def send_alert(self, message, frame=None):
        msg = MIMEText(f"Alerta: objeto perigoso detectado ({message})!")
        msg['Subject'] = "Alerta de Segurança"
        msg['From'] =  self.from_email
        msg['To'] = self.recipient_email

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            #smtp.starttls()
            smtp.login(self.from_email, self.password)
            smtp.sendmail(self.from_email, self.recipient_email, msg.as_string())

def process_frame(frame, model, alert_handler: Alert, timestamp: str = ""):
    results = model(frame)
    boxes = results[0].boxes

    for box in boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        conf = float(box.conf[0])
        threshold = CLASS_THRESHOLDS.get(label, 0.6)

        if conf > threshold:
            message = f"Detectado: {label} com confiança {conf:.2f}"
            if timestamp:
                message += f" em {timestamp}"
            logger.debug(message)
            alert_handler.send_alert(message)

    img_bgr = cv2.cvtColor(results[0].plot(), cv2.COLOR_RGB2BGR)
    return img_bgr

def process_video(video_path: str, model, alert_handler: Alert):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Erro ao abrir vídeo: {video_path}")
        return

    cv2.namedWindow("Detecção", cv2.WINDOW_NORMAL)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        current_time_sec = current_time_ms / 1000
        timestamp = f"{int(current_time_sec // 3600):02}:{int((current_time_sec % 3600) // 60):02}:{int(current_time_sec % 60):02}"

        processed_frame = process_frame(frame, model, alert_handler, timestamp)
        cv2.imshow("Detecção", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

def process_image(image_path: str, model, alert_handler: Alert):
    frame = cv2.imread(image_path)
    if frame is None:
        logger.error(f"Erro ao carregar imagem: {image_path}")
        return
    processed_frame = process_frame(frame, model, alert_handler)
    cv2.namedWindow("Detecção", cv2.WINDOW_NORMAL)
    cv2.imshow("Detecção", processed_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python inference.py caminho/do/arquivo.(mp4|jpg|png)")
        sys.exit(1)

    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        logger.error(f"Arquivo não encontrado: {file_path}")
        sys.exit(1)

    alert = ConsoleAlert()
    #alert = EmailAlert(recipient_email="rivervaladao@gmail.com")
    
    model = YOLO(path_model)

    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".mp4", ".avi", ".mov", ".mkv"]:
        process_video(file_path, model, alert)
    elif ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        process_image(file_path, model, alert)
    else:
        logger.error("Formato de arquivo não suportado.")