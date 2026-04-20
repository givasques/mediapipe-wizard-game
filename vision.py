from pathlib import Path
from urllib.request import urlretrieve

import cv2
import mediapipe as mp


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")


def open_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro: nao foi possivel acessar a webcam.")
        return None
    return cap


def ensure_model():
    if MODEL_PATH.exists():
        return MODEL_PATH

    print("Baixando o modelo do MediaPipe...")
    try:
        urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception as exc:
        print("Erro: nao foi possivel baixar o modelo do MediaPipe.")
        print(f"Baixe manualmente em: {MODEL_URL}")
        print(f"Salve o arquivo em: {MODEL_PATH}")
        raise exc

    return MODEL_PATH


def create_hand_landmarker():
    model_path = ensure_model()
    base_options = mp.tasks.BaseOptions(model_asset_buffer=model_path.read_bytes())
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def capture_frame(cap):
    ret, frame = cap.read()
    if not ret:
        return None
    return cv2.flip(frame, 1)


def frame_to_mp_image(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)


def draw_landmarks_on_frame(frame, detection_result):
    mp_drawing = mp.tasks.vision.drawing_utils
    mp_styles = mp.tasks.vision.drawing_styles
    mp_connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

    for hand_landmarks, handedness in zip(
        detection_result.hand_landmarks, detection_result.handedness
    ):
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_connections,
            mp_styles.get_default_hand_landmarks_style(),
            mp_styles.get_default_hand_connections_style(),
        )

        h, w, _ = frame.shape
        xs = [landmark.x for landmark in hand_landmarks]
        ys = [landmark.y for landmark in hand_landmarks]
        x = int(min(xs) * w)
        y = int(min(ys) * h) - 10
        label_map = {"Left": "Direita", "Right": "Esquerda"}
        raw_label = handedness[0].category_name
        label = label_map.get(raw_label, raw_label)

        cv2.putText(
            frame,
            label,
            (x, max(y, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
