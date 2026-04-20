import math

def get_hand_side(handedness_label):
    if handedness_label == "Left":
        return "Direita"
    if handedness_label == "Right":
        return "Esquerda"
    return handedness_label


def _finger_is_up(hand_landmarks, tip_index, pip_index):
    return hand_landmarks[tip_index].y < hand_landmarks[pip_index].y


def _finger_vertical_state(hand_landmarks, tip_index, reference_index, margin=0.03):
    tip = hand_landmarks[tip_index]
    reference = hand_landmarks[reference_index]
    delta = tip.y - reference.y

    if delta < -margin:
        return "up"
    if delta > margin:
        return "down"
    return "neutral"


def _thumb_extended(hand_landmarks, margin=0.02):
    wrist = hand_landmarks[0]
    thumb_mcp = hand_landmarks[2]
    thumb_tip = hand_landmarks[4]

    thumb_tip_distance = math.hypot(thumb_tip.x - wrist.x, thumb_tip.y - wrist.y)
    thumb_mcp_distance = math.hypot(thumb_mcp.x - wrist.x, thumb_mcp.y - wrist.y)
    return thumb_tip_distance > thumb_mcp_distance + margin


def _finger_tip_index(finger_name):
    return {
        "index": 8,
        "thumb": 4,
        "pinky": 20,
    }.get(finger_name)


def _is_touching_wrist(hand_landmarks, finger_name, wrist_landmark, threshold=0.10):
    tip_index = _finger_tip_index(finger_name)
    if tip_index is None or wrist_landmark is None:
        return False

    tip = hand_landmarks[tip_index]
    dx = tip.x - wrist_landmark.x
    dy = tip.y - wrist_landmark.y
    return math.hypot(dx, dy) <= threshold


def classify_right_hand_gesture(hand_landmarks):
    index_up = _finger_vertical_state(hand_landmarks, 8, 6) == "up"
    middle_up = _finger_vertical_state(hand_landmarks, 12, 10) == "up"
    ring_up = _finger_vertical_state(hand_landmarks, 16, 14) == "up"
    pinky_up = _finger_vertical_state(hand_landmarks, 20, 18) == "up"

    thumb_tip = hand_landmarks[4]
    thumb_ip = hand_landmarks[3]
    thumb_side = abs(thumb_tip.x - thumb_ip.x) > 0.04 and thumb_tip.y < hand_landmarks[2].y + 0.18

    if index_up and thumb_side and not middle_up and not ring_up and not pinky_up:
        return {
            "mode": "PODER",
            "combo": "indicador + dedao",
            "element": "TERRA",
            "action": "TERRA CASTED",
        }

    if index_up and middle_up and not ring_up and not pinky_up:
        return {
            "mode": "PODER",
            "combo": "sinal de paz",
            "element": "AGUA",
            "action": "AGUA CASTED",
        }

    if index_up and not middle_up and not ring_up and not pinky_up:
        return {
            "mode": "PODER",
            "combo": "indicador",
            "element": "FOGO",
            "action": "FOGO CASTED",
        }

    return {
        "mode": "NEUTRO",
        "combo": None,
        "element": None,
        "action": "SEM COMBO",
    }


def classify_left_hand_target(hand_landmarks, frame_width):
    wrist = hand_landmarks[0]
    index_tip = hand_landmarks[8]

    # valor do dx indica se o indicador está para a esquerda ou direita do punho
    # valor do dy indica se a ponta do dedo está mais para baixo ou não
    # dy é necessário pois queremos saber se o dedo está apontando para cima também
    dx = index_tip.x - wrist.x
    dy = index_tip.y - wrist.y

    # converte a direção do dedo em ângulo
    angle = math.degrees(math.atan2(dx, -dy))

    # Dentro da margem de 15º consideramos que o dedo está apontando para cima
    # Limite de tolerância
    if angle < -15:
        return {
            "zone": "1/3",
            "action": "Mira no grupo esquerdo",
            "angle": angle,
        }

    if angle > 15:
        return {
            "zone": "3/3",
            "action": "Mira no grupo direito",
            "angle": angle,
        }

    return {
        "zone": "2/3",
        "action": "Mira no centro",
        "angle": angle,
    }


def summarize_hand(hand_side, gesture_data):
    return f"{hand_side}: {gesture_data['action']}"
