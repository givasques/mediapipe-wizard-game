import math
import random
from pathlib import Path

import cv2
import numpy as np

from game_logic import classify_left_hand_target, classify_right_hand_gesture, get_hand_side
from vision import (
    create_hand_landmarker,
    capture_frame,
    draw_landmarks_on_frame,
    frame_to_mp_image,
    open_camera,
)


ASSETS_DIR = Path(__file__).with_name("images")
GAME_VIEW_SIZE = (1280, 720)
WEBCAM_VIEW_SIZE = (420, 236)
HEART_SPRITE_PATH = ASSETS_DIR / "etc" / "heart.png"
BACKGROUND_SPRITE_PATH = ASSETS_DIR / "etc" / "background.png"
LEGEND_SPRITE_PATH = ASSETS_DIR / "etc" / "legenda.png"
PLAYER_SPRITE_PATH = ASSETS_DIR / "wizard_main" / "wizard.png"
HIGH_SCORE_PATH = Path(__file__).with_name("high_score.txt")
MAGE_FOLDER_BY_ELEMENT = {
    "FOGO": "wizard_fire",
    "GELO": "wizard_ice",
    "ELETRICO": "wizard_electric",
    None: "wizard",
}
MAGE_STATE_PREFIX = {
    "idle": "1_IDLE",
    "walk": "2_WALK",
    "attack": "5_ATTACK",
    "hurt": "6_HURT",
    "die": "7_DIE",
}


def _draw_panel(frame, x, title, lines, align="left"):
    h, w, _ = frame.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    title_scale = 0.5
    body_scale = 0.45
    title_color = (255, 255, 255)
    body_color = (230, 230, 230)

    y = 25
    padding = 10
    line_step = 22

    all_lines = [title] + lines
    max_width = 0
    for idx, text in enumerate(all_lines):
        scale = title_scale if idx == 0 else body_scale
        thickness = 2 if idx == 0 else 1
        (text_width, _text_height), _ = cv2.getTextSize(text, font, scale, thickness)
        max_width = max(max_width, text_width)

    box_width = max_width + padding * 2
    box_height = 18 + len(all_lines) * line_step

    if align == "right":
        x1 = w - x - box_width
    else:
        x1 = x
    y1 = y
    x2 = x1 + box_width
    y2 = y1 + box_height

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

    cv2.putText(frame, title, (x1 + padding, y1 + 18), font, title_scale, title_color, 2)

    text_y = y1 + 18 + line_step
    for line in lines:
        cv2.putText(frame, line, (x1 + padding, text_y), font, body_scale, body_color, 1)
        text_y += line_step


def _load_high_score():
    try:
        content = HIGH_SCORE_PATH.read_text(encoding="utf-8").strip()
        return max(0, int(content)) if content else 0
    except (OSError, ValueError):
        return 0


def _save_high_score(value):
    try:
        HIGH_SCORE_PATH.write_text(str(max(0, int(value))), encoding="utf-8")
    except OSError:
        pass


def _difficulty_from_defeats(defeats):
    if defeats >= 30:
        return 4
    if defeats >= 20:
        return 3
    if defeats >= 10:
        return 2
    return 1


def _difficulty_profile(level):
    profiles = {
        1: {"attack_start": (110, 170), "attack_loop": (160, 240), "projectile_speed": (2.9, 3.6), "drop_speed": (2.8, 3.6)},
        2: {"attack_start": (90, 150), "attack_loop": (140, 210), "projectile_speed": (3.2, 4.0), "drop_speed": (3.1, 4.0)},
        3: {"attack_start": (75, 130), "attack_loop": (120, 180), "projectile_speed": (3.6, 4.5), "drop_speed": (3.4, 4.4)},
        4: {"attack_start": (60, 110), "attack_loop": (95, 150), "projectile_speed": (4.0, 5.0), "drop_speed": (3.8, 4.8)},
    }
    return profiles.get(level, profiles[4])


def _draw_game_hud(scene, player_state):
    lines = [
        f"Pontos: {player_state['score']}",
        f"Recorde: {player_state['high_score']}",
        f"Nivel: {player_state['difficulty']}",
    ]
    _draw_panel(scene, 12, "STATUS", lines, align="left")


def _point_in_rect(x, y, rect):
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


def _draw_game_over_screen(scene, player_state):
    h, w, _ = scene.shape
    overlay = scene.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.62, scene, 0.38, 0, scene)

    title = "VOCE PERDEU"
    subtitle = f"Pontos: {player_state['score']} | Recorde: {player_state['high_score']}"
    hint = "Clique em JOGAR NOVAMENTE"
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(scene, title, (w // 2 - 150, h // 2 - 40), font, 1.2, (255, 255, 255), 3)
    cv2.putText(scene, subtitle, (w // 2 - 175, h // 2 + 5), font, 0.7, (230, 230, 230), 2)
    cv2.putText(scene, hint, (w // 2 - 160, h // 2 + 45), font, 0.55, (210, 210, 210), 1)

    button_w = 280
    button_h = 58
    x1 = w // 2 - button_w // 2
    y1 = h // 2 + 90
    x2 = x1 + button_w
    y2 = y1 + button_h

    cv2.rectangle(scene, (x1, y1), (x2, y2), (255, 255, 255), 2)
    cv2.rectangle(scene, (x1 + 2, y1 + 2), (x2 - 2, y2 - 2), (25, 25, 40), -1)
    cv2.putText(scene, "JOGAR NOVAMENTE", (x1 + 22, y1 + 37), font, 0.75, (255, 255, 255), 2)
    return (x1, y1, x2, y2)


def _draw_start_screen(scene):
    h, w, _ = scene.shape
    overlay = scene.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.62, scene, 0.38, 0, scene)

    title = "MAGIC GAME"
    subtitle = "Use as maos para mirar, atacar e defender"
    hint = "Clique em INICIAR para comecar"
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(scene, title, (w // 2 - 145, h // 2 - 50), font, 1.2, (255, 255, 255), 3)
    cv2.putText(scene, subtitle, (w // 2 - 205, h // 2 - 5), font, 0.65, (230, 230, 230), 2)
    cv2.putText(scene, hint, (w // 2 - 170, h // 2 + 35), font, 0.55, (210, 210, 210), 1)

    button_w = 220
    button_h = 58
    x1 = w // 2 - button_w // 2
    y1 = h // 2 + 85
    x2 = x1 + button_w
    y2 = y1 + button_h

    cv2.rectangle(scene, (x1, y1), (x2, y2), (255, 255, 255), 2)
    cv2.rectangle(scene, (x1 + 2, y1 + 2), (x2 - 2, y2 - 2), (25, 25, 40), -1)
    cv2.putText(scene, "INICIAR", (x1 + 57, y1 + 37), font, 0.75, (255, 255, 255), 2)
    return (x1, y1, x2, y2)


def _reset_match_state(player_state, scene_width):
    player_state["lives"] = 5
    player_state["score"] = 0
    player_state["defeated_enemies"] = 0
    player_state["difficulty"] = 1

    enemies = _create_enemies(scene_width, player_state["difficulty"])
    projectiles = []
    enemy_projectiles = []
    impact_effects = []
    feedback = None
    feedback_timer = 0
    cast_armed = True
    player_action_timer = 0
    return enemies, projectiles, enemy_projectiles, impact_effects, feedback, feedback_timer, cast_armed, player_action_timer


def _crop_transparent_border(image):
    if image is None or image.ndim != 3 or image.shape[2] != 4:
        return image

    alpha = image[:, :, 3]
    ys, xs = np.where(alpha > 0)
    if len(xs) == 0 or len(ys) == 0:
        return image

    x1, x2 = xs.min(), xs.max() + 1
    y1, y2 = ys.min(), ys.max() + 1
    return image[y1:y2, x1:x2]


def _load_sprite_frame(path):
    try:
        raw_bytes = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if raw_bytes.size == 0:
        return None

    sprite = cv2.imdecode(raw_bytes, cv2.IMREAD_UNCHANGED)
    if sprite is None:
        return None
    if sprite.ndim == 3 and sprite.shape[2] == 3:
        alpha = np.full((sprite.shape[0], sprite.shape[1], 1), 255, dtype=np.uint8)
        sprite = np.concatenate([sprite, alpha], axis=2)
    return _crop_transparent_border(sprite)


def _load_sprite_animation(folder, prefix):
    frames = []
    if not folder.exists():
        return frames

    for path in sorted(folder.glob(f"{prefix}_*.png")):
        sprite = _load_sprite_frame(path)
        if sprite is not None:
            frames.append(sprite)
    return frames


def _load_mage_sprites():
    sprites = {}
    for element, folder_name in MAGE_FOLDER_BY_ELEMENT.items():
        folder = ASSETS_DIR / folder_name
        sprites[element] = {
            state: _load_sprite_animation(folder, prefix)
            for state, prefix in MAGE_STATE_PREFIX.items()
        }
    return sprites


def _load_heart_sprite():
    return _load_sprite_frame(HEART_SPRITE_PATH)


def _load_background_sprite():
    return _load_sprite_frame(BACKGROUND_SPRITE_PATH)


def _load_legend_sprite():
    return _load_sprite_frame(LEGEND_SPRITE_PATH)


def _load_player_sprite():
    return _load_sprite_frame(PLAYER_SPRITE_PATH)


def _resize_sprite_to_height(sprite, target_height):
    if sprite is None:
        return None

    sprite_h, sprite_w = sprite.shape[:2]
    if sprite_h <= 0 or sprite_w <= 0:
        return None

    scale = target_height / float(sprite_h)
    target_w = max(1, int(sprite_w * scale))
    return cv2.resize(sprite, (target_w, target_height), interpolation=cv2.INTER_AREA)


def _prepare_sprite_library(sprite_library, target_height):
    prepared = {}
    for element, states in sprite_library.items():
        prepared[element] = {}
        for state, frames in states.items():
            prepared[element][state] = [
                _resize_sprite_to_height(frame, target_height)
                for frame in frames
                if frame is not None
            ]
    return prepared


def _prepare_background(background_sprite, target_size):
    if background_sprite is None:
        return None

    target_w, target_h = target_size
    bg_h, bg_w = background_sprite.shape[:2]
    if bg_h <= 0 or bg_w <= 0:
        return None

    scale = max(target_w / float(bg_w), target_h / float(bg_h))
    scaled_w = max(1, int(bg_w * scale))
    scaled_h = max(1, int(bg_h * scale))
    resized = cv2.resize(background_sprite, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)

    x_offset = int(max(0, scaled_w - target_w) * 0.05)
    x1 = max(0, min(scaled_w - target_w, (scaled_w - target_w) // 2 + x_offset))
    y1 = max(0, (scaled_h - target_h) // 2)
    cropped = resized[y1:y1 + target_h, x1:x1 + target_w]
    if cropped.shape[0] != target_h or cropped.shape[1] != target_w:
        cropped = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_AREA)
    if cropped.ndim == 3 and cropped.shape[2] == 4:
        bgr = cropped[:, :, :3].astype(np.float32)
        alpha = cropped[:, :, 3:4].astype(np.float32) / 255.0
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        blended = (canvas.astype(np.float32) * (1.0 - alpha) + bgr * alpha).astype(np.uint8)
        return blended
    return cropped[:, :, :3].copy()


def _pick_sprite_frame(sprite_library, element, state, frame_index):
    element_key = element if element in sprite_library else None
    state_frames = sprite_library.get(element_key, {}).get(state, [])
    if not state_frames:
        state_frames = sprite_library.get(element_key, {}).get("idle", [])
    if not state_frames:
        return None
    return state_frames[frame_index % len(state_frames)]


def _overlay_prepared_sprite(scene, sprite, anchor_x, anchor_y):
    if sprite is None:
        return

    sprite_h, sprite_w = sprite.shape[:2]
    if sprite_h <= 0 or sprite_w <= 0:
        return

    x1 = int(anchor_x - sprite_w // 2)
    y2 = int(anchor_y)
    y1 = int(y2 - sprite_h)

    scene_h, scene_w = scene.shape[:2]
    sx1 = max(0, x1)
    sy1 = max(0, y1)
    sx2 = min(scene_w, x1 + sprite_w)
    sy2 = min(scene_h, y1 + sprite_h)
    if sx1 >= sx2 or sy1 >= sy2:
        return

    rx1 = sx1 - x1
    ry1 = sy1 - y1
    rx2 = rx1 + (sx2 - sx1)
    ry2 = ry1 + (sy2 - sy1)

    sprite_roi = sprite[ry1:ry2, rx1:rx2]
    bgr = sprite_roi[:, :, :3].astype(np.float32)
    alpha = sprite_roi[:, :, 3:4].astype(np.float32) / 255.0

    scene_roi = scene[sy1:sy2, sx1:sx2].astype(np.float32)
    blended = scene_roi * (1.0 - alpha) + bgr * alpha
    scene[sy1:sy2, sx1:sx2] = blended.astype(np.uint8)


def _tint_sprite(sprite, brightness=0.45):
    if sprite is None:
        return None

    tinted = sprite.copy()
    if tinted.ndim != 3 or tinted.shape[2] < 3:
        return tinted

    tinted[:, :, :3] = np.clip(tinted[:, :, :3].astype(np.float32) * brightness, 0, 255).astype(np.uint8)
    return tinted


def _element_color(element):
    return {
        "FOGO": (40, 80, 255),
        "GELO": (255, 180, 80),
        "ELETRICO": (180, 60, 255),
        "AGUA": (255, 140, 60),
        "TERRA": (80, 180, 80),
        None: (200, 200, 200),
    }.get(element, (200, 200, 200))


def _projectile_color(element):
    return _element_color(element)


def _mix_color(color_a, color_b, ratio):
    ratio = max(0.0, min(1.0, ratio))
    return tuple(int(color_a[i] * (1.0 - ratio) + color_b[i] * ratio) for i in range(3))


def _enemy_display_color(enemy):
    if not enemy["alive"]:
        return (85, 85, 95)

    if enemy["lives"] >= 2:
        return enemy["color"]

    wounded_palette = {
        "FOGO": (255, 165, 120),
        "GELO": (120, 200, 255),
        "ELETRICO": (210, 120, 255),
    }
    return wounded_palette.get(enemy["element"], _mix_color(enemy["color"], (255, 255, 255), 0.35))


def _enemy_weakness(enemy_element):
    return {
        "FOGO": "AGUA",
        "GELO": "FOGO",
        "ELETRICO": "TERRA",
    }.get(enemy_element)


def _enemy_positions(scene_width):
    return [int(scene_width * 0.2), int(scene_width * 0.5), int(scene_width * 0.8)]


def _random_enemy_template():
    return random.choice(
        [
            {"element": "FOGO", "color": (40, 80, 255)},
            {"element": "GELO", "color": (120, 200, 255)},
            {"element": "ELETRICO", "color": (210, 120, 255)},
        ]
    ).copy()


def _build_enemy(slot_x, target_y, start_at_top=False, difficulty_level=1):
    template = _random_enemy_template()
    profile = _difficulty_profile(difficulty_level)
    y = -60 if start_at_top else target_y
    return {
        "element": template["element"],
        "weak_to": _enemy_weakness(template["element"]),
        "color": template["color"],
        "x": slot_x,
        "y": y,
        "target_y": target_y,
        "alive": True,
        "lives": 2,
        "max_lives": 2,
        "hit_flash": 0,
        "hit_color": None,
        "drop_speed": random.uniform(*profile["drop_speed"]),
        "arrived": not start_at_top,
        "attack_cooldown": random.randint(*profile["attack_start"]),
    }


def _create_enemies(scene_width, difficulty_level=1):
    positions = _enemy_positions(scene_width)
    return [_build_enemy(slot_x, 90, start_at_top=False, difficulty_level=difficulty_level) for slot_x in positions]


def _draw_player_lives(scene, lives, max_lives, heart_sprite):
    heart_gap = 4
    heart_size = 14
    start_x = 20
    y = scene.shape[0] - 22
    filled = heart_sprite
    empty = _tint_sprite(heart_sprite, 0.25) if heart_sprite is not None else None

    for idx in range(max_lives):
        x = start_x + idx * (heart_size + heart_gap)
        if heart_sprite is not None:
            sprite = filled if idx < lives else empty
            _overlay_prepared_sprite(scene, sprite, x, y)
        else:
            color = (0, 0, 255) if idx < lives else (60, 60, 60)
            cv2.circle(scene, (x, y), 5, color, -1)


def _draw_enemies(scene, enemies, mage_sprites, frame_index, heart_sprite):
    for enemy in enemies:
        _draw_enemy_sprite(scene, enemy, mage_sprites, frame_index)
        center = (int(enemy["x"]), int(enemy["y"]))
        heart_y = int(enemy["y"] + 18)
        heart_gap = 4
        heart_size = 14
        total_width = heart_size * 2 + heart_gap
        start_x = int(center[0] - total_width // 2 + heart_size // 2)

        if heart_sprite is not None:
            filled = heart_sprite
            empty = _tint_sprite(heart_sprite, 0.25)
            for idx in range(enemy["max_lives"]):
                heart_x = start_x + idx * (heart_size + heart_gap)
                sprite = filled if idx < enemy["lives"] else empty
                _overlay_prepared_sprite(scene, sprite, heart_x, heart_y)
        else:
            for idx in range(enemy["max_lives"]):
                heart_x = start_x + idx * (heart_size + heart_gap)
                color = (0, 0, 255) if idx < enemy["lives"] else (60, 60, 60)
                cv2.circle(scene, (heart_x, heart_y), 5, color, -1)


def _enemy_sprite_state(enemy):
    if not enemy["alive"]:
        return "die"
    if not enemy.get("arrived", True):
        return "walk"
    if enemy.get("hit_flash", 0) > 0:
        return "hurt"
    return "idle"


def _draw_enemy_sprite(scene, enemy, mage_sprites, frame_index):
    sprite = _pick_sprite_frame(mage_sprites, enemy["element"], _enemy_sprite_state(enemy), frame_index)
    if sprite is None:
        color = _enemy_display_color(enemy)
        outline_color = (255, 255, 255)
        if enemy.get("hit_flash", 0) > 0 and enemy.get("hit_color") is not None:
            color = enemy["hit_color"]
            outline_color = enemy["hit_color"]
        center = (int(enemy["x"]), int(enemy["y"]))
        cv2.circle(scene, center, 30, (25, 25, 35), -1)
        cv2.circle(scene, center, 28, color, -1)
        cv2.circle(scene, center, 28, outline_color, 2)
        return

    anchor_x = int(enemy["x"])
    anchor_y = int(enemy["y"] + 110)
    _overlay_prepared_sprite(scene, sprite, anchor_x, anchor_y)


def _spawn_projectile(player_center, element, target_enemy_index, enemies):
    if target_enemy_index is None or target_enemy_index < 0 or target_enemy_index >= len(enemies):
        return None
    target = enemies[target_enemy_index]
    if not target["alive"] or not target.get("arrived", True):
        return None
    return {
        "element": element,
        "x": float(player_center[0]),
        "y": float(player_center[1] - 20),
        "prev_x": float(player_center[0]),
        "prev_y": float(player_center[1] - 20),
        "target_index": target_enemy_index,
        "speed": 11.0,
        "done": False,
        "trail": [(float(player_center[0]), float(player_center[1] - 20))],
    }


def _spawn_enemy_projectile(enemy, player_center, difficulty_level=1):
    dx = player_center[0] - enemy["x"]
    dy = player_center[1] - enemy["y"]
    distance = math.hypot(dx, dy)
    if distance <= 0:
        distance = 1.0
    profile = _difficulty_profile(difficulty_level)
    speed = random.uniform(*profile["projectile_speed"])
    return {
        "kind": "enemy",
        "element": enemy["element"],
        "x": float(enemy["x"]),
        "y": float(enemy["y"] + 20),
        "prev_x": float(enemy["x"]),
        "prev_y": float(enemy["y"] + 20),
        "target_x": float(player_center[0]),
        "target_y": float(player_center[1]),
        "speed": speed,
        "done": False,
        "trail": [(float(enemy["x"]), float(enemy["y"] + 20))],
    }


def _append_impact(impact_effects, x, y, color):
    impact_effects.append(
        {
            "x": float(x),
            "y": float(y),
            "color": color,
            "ttl": 14,
            "max_ttl": 14,
        }
    )


def _update_projectiles(projectiles, enemies, impact_effects, player_state):
    feedback = None
    remaining = []

    for projectile in projectiles:
        if projectile["done"]:
            continue

        target = enemies[projectile["target_index"]]
        if not target["alive"] or not target.get("arrived", True):
            continue

        target_point = (target["x"], target["y"])
        dx = target_point[0] - projectile["x"]
        dy = target_point[1] - projectile["y"]
        distance = math.hypot(dx, dy)

        if distance <= 18:
            projectile["done"] = True
            if projectile["element"] == target["weak_to"]:
                target["lives"] = max(0, target["lives"] - 1)
                target["hit_flash"] = 18
                target["hit_color"] = (80, 240, 120) if target["lives"] > 0 else (60, 220, 60)
                _append_impact(impact_effects, projectile["x"], projectile["y"], target["hit_color"])
                player_state["score"] += 15
                if target["lives"] <= 0:
                    player_state["defeated_enemies"] += 1
                if target["lives"] <= 0:
                    target["alive"] = False
                    target["arrived"] = False
                    target["spawn_delay"] = 8
                    feedback = {
                        "text": f"DERROTOU: {projectile['element']} > {target['element']}",
                        "color": (80, 220, 80),
                    }
                else:
                    feedback = {
                        "text": f"ACERTO CORRETO: {projectile['element']} > {target['element']} | vidas: {target['lives']}/{target['max_lives']}",
                        "color": (80, 220, 80),
                    }
            else:
                target["hit_flash"] = 18
                target["hit_color"] = (60, 60, 255)
                _append_impact(impact_effects, projectile["x"], projectile["y"], target["hit_color"])
                player_state["score"] = max(0, player_state["score"] - 10)
                feedback = {
                    "text": f"TIPO ERRADO: {projectile['element']} nao derrota {target['element']} | vidas: {target['lives']}/{target['max_lives']}",
                    "color": (60, 60, 255),
                }
            continue

        if distance > 0:
            projectile["prev_x"] = projectile["x"]
            projectile["prev_y"] = projectile["y"]
            step_x = (dx / distance) * projectile["speed"]
            step_y = (dy / distance) * projectile["speed"]
            projectile["x"] += step_x
            projectile["y"] += step_y
            projectile["trail"].append((projectile["x"], projectile["y"]))
            if len(projectile["trail"]) > 8:
                projectile["trail"] = projectile["trail"][-8:]

        remaining.append(projectile)

    return remaining, feedback


def _update_enemy_projectiles(enemy_projectiles, player_center, player_state, impact_effects, shield_active):
    remaining = []
    feedback = None

    for projectile in enemy_projectiles:
        if projectile["done"]:
            continue

        dx = projectile["target_x"] - projectile["x"]
        dy = projectile["target_y"] - projectile["y"]
        distance = math.hypot(dx, dy)
        if distance <= 16:
            projectile["done"] = True
            hit_color = _projectile_color(projectile["element"])
            _append_impact(impact_effects, projectile["x"], projectile["y"], hit_color)
            if shield_active:
                feedback = {
                    "text": f"ESCUDO BLOQUEOU {projectile['element']}",
                    "color": (80, 190, 255),
                }
            else:
                player_state["lives"] = max(0, player_state["lives"] - 1)
                feedback = {
                    "text": f"VOCE LEVOU {projectile['element']} | vidas: {player_state['lives']}/{player_state['max_lives']}",
                    "color": (255, 110, 110),
                }
            continue

        if distance > 0:
            projectile["prev_x"] = projectile["x"]
            projectile["prev_y"] = projectile["y"]
            projectile["x"] += (dx / distance) * projectile["speed"]
            projectile["y"] += (dy / distance) * projectile["speed"]
            projectile["trail"].append((projectile["x"], projectile["y"]))
            if len(projectile["trail"]) > 8:
                projectile["trail"] = projectile["trail"][-8:]

        remaining.append(projectile)

    return remaining, feedback


def _tick_enemy_flashes(enemies):
    for enemy in enemies:
        if enemy.get("hit_flash", 0) > 0:
            enemy["hit_flash"] -= 1
            if enemy["hit_flash"] <= 0:
                enemy["hit_color"] = None


def _update_enemy_respawns(enemies, difficulty_level):
    profile = _difficulty_profile(difficulty_level)
    for enemy in enemies:
        if enemy.get("alive") and not enemy.get("arrived", True):
            enemy["y"] = min(enemy["target_y"], enemy["y"] + enemy.get("drop_speed", 2.0))
            if enemy["y"] >= enemy["target_y"]:
                enemy["y"] = enemy["target_y"]
                enemy["arrived"] = True
                enemy["attack_cooldown"] = random.randint(*profile["attack_start"])
            continue

        if not enemy.get("alive") and enemy.get("spawn_delay", 0) > 0:
            enemy["spawn_delay"] -= 1
            if enemy["spawn_delay"] <= 0:
                template = _random_enemy_template()
                enemy["element"] = template["element"]
                enemy["weak_to"] = _enemy_weakness(template["element"])
                enemy["color"] = template["color"]
                enemy["lives"] = 2
                enemy["max_lives"] = 2
                enemy["alive"] = True
                enemy["hit_flash"] = 0
                enemy["hit_color"] = None
                enemy["y"] = -60
                enemy["drop_speed"] = random.uniform(*profile["drop_speed"])
                enemy["arrived"] = False
                enemy["spawn_delay"] = 0
                enemy["attack_cooldown"] = random.randint(*profile["attack_start"])


def _update_enemy_attacks(enemies, enemy_projectiles, player_center, difficulty_level):
    profile = _difficulty_profile(difficulty_level)
    for enemy in enemies:
        if not enemy.get("alive") or not enemy.get("arrived", True):
            continue

        if "attack_cooldown" not in enemy:
            enemy["attack_cooldown"] = random.randint(*profile["attack_start"])

        enemy["attack_cooldown"] -= 1
        if enemy["attack_cooldown"] <= 0:
            enemy_projectiles.append(_spawn_enemy_projectile(enemy, player_center, difficulty_level))
            enemy["attack_cooldown"] = random.randint(*profile["attack_loop"])


def _update_impact_effects(impact_effects):
    remaining = []
    for impact in impact_effects:
        impact["ttl"] -= 1
        if impact["ttl"] > 0:
            remaining.append(impact)
    return remaining


def _draw_player(scene, center, active_element):
    x, y = center
    body_color = (210, 210, 210)
    hat_color = (100, 100, 100)
    accent_color = _element_color(active_element)

    cv2.circle(scene, (x, y), 32, body_color, -1)
    cv2.circle(scene, (x, y), 34, (80, 80, 80), 2)
    cv2.rectangle(scene, (x - 18, y - 52), (x + 18, y - 22), hat_color, -1)
    pts = np.array([[x - 26, y - 22], [x, y - 72], [x + 26, y - 22]], np.int32)
    cv2.fillConvexPoly(scene, pts, hat_color)

    cv2.circle(scene, (x, y + 2), 14, accent_color, -1)
    cv2.circle(scene, (x, y + 2), 14, (255, 255, 255), 2)

    if shield_active:
        cv2.ellipse(scene, (x, y), (58, 58), 0, 200, 340, (0, 220, 255), 4)
        cv2.ellipse(scene, (x, y), (58, 58), 0, 20, 160, (0, 220, 255), 2)


def _draw_player_sprite(scene, center, mage_sprites, active_element, action_state, frame_index):
    x, y = center
    sprite = _pick_sprite_frame(mage_sprites, None, action_state, frame_index)

    if sprite is None:
        _draw_player(scene, center, active_element)
        return

    _overlay_prepared_sprite(scene, sprite, x, y + 35)


def _draw_shield_dome(scene, center):
    x, y = center
    overlay = scene.copy()
    dome_color = (255, 180, 60)
    cv2.ellipse(overlay, (x, y - 6), (96, 96), 0, 0, 360, dome_color, -1)
    cv2.addWeighted(overlay, 0.22, scene, 0.78, 0, scene)
    cv2.ellipse(scene, (x, y - 6), (96, 96), 0, 0, 360, (255, 220, 120), 2)
    cv2.ellipse(scene, (x, y - 6), (82, 82), 0, 0, 360, (120, 200, 255), 1)


def _draw_aim(scene, player_center, target_zone):
    x, y = player_center
    start_point = (x, y - 14)
    direction_targets = {
        "1/3": (int(scene.shape[1] * 0.22), 175),
        "2/3": (int(scene.shape[1] * 0.50), 195),
        "3/3": (int(scene.shape[1] * 0.78), 175),
    }
    direction_point = direction_targets.get(target_zone, direction_targets["2/3"])
    dx = direction_point[0] - start_point[0]
    dy = direction_point[1] - start_point[1]
    length = max(1.0, math.hypot(dx, dy))
    desired_length = 70.0
    scale = desired_length / length
    end_point = (
        int(start_point[0] + dx * scale),
        int(start_point[1] + dy * scale),
    )

    segments = 4
    for idx in range(segments):
        if idx % 2 == 0:
            p1_x = int(start_point[0] + (end_point[0] - start_point[0]) * (idx / segments))
            p1_y = int(start_point[1] + (end_point[1] - start_point[1]) * (idx / segments))
            p2_x = int(start_point[0] + (end_point[0] - start_point[0]) * ((idx + 1) / segments))
            p2_y = int(start_point[1] + (end_point[1] - start_point[1]) * ((idx + 1) / segments))
            cv2.line(scene, (p1_x, p1_y), (p2_x, p2_y), (220, 220, 220), 1)

    ux = dx / length
    uy = dy / length
    arrow_length = 7
    arrow_width = 4
    tip = end_point
    left = (
        int(tip[0] - ux * arrow_length - uy * arrow_width),
        int(tip[1] - uy * arrow_length + ux * arrow_width),
    )
    right = (
        int(tip[0] - ux * arrow_length + uy * arrow_width),
        int(tip[1] - uy * arrow_length - ux * arrow_width),
    )
    cv2.line(scene, left, tip, (220, 220, 220), 1)
    cv2.line(scene, right, tip, (220, 220, 220), 1)


def _draw_feedback_toast(scene, feedback):
    if not feedback:
        return

    h, w, _ = scene.shape
    text = feedback["text"]
    color = feedback["color"]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    box_w = text_width + 26
    box_h = text_height + baseline + 18
    x1 = max(20, (w - box_w) // 2)
    y1 = 20
    x2 = x1 + box_w
    y2 = y1 + box_h

    overlay = scene.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, scene, 0.45, 0, scene)
    cv2.rectangle(scene, (x1, y1), (x2, y2), color, 1)
    cv2.putText(scene, text, (x1 + 13, y1 + text_height + 8), font, scale, color, thickness)


def _draw_projectiles(scene, projectiles):
    for projectile in projectiles:
        color = _projectile_color(projectile["element"])
        trail = projectile.get("trail", [])
        if len(trail) >= 2:
            for idx in range(1, len(trail)):
                p1 = trail[idx - 1]
                p2 = trail[idx]
                thickness = max(1, idx // 2 + 1)
                fade = idx / max(1, len(trail) - 1)
                trail_color = _mix_color(color, (255, 255, 255), 0.2 * (1.0 - fade))
                cv2.line(
                    scene,
                    (int(p1[0]), int(p1[1])),
                    (int(p2[0]), int(p2[1])),
                    trail_color,
                    thickness,
                )

        center = (int(projectile["x"]), int(projectile["y"]))
        cv2.circle(scene, center, 13, _mix_color(color, (255, 255, 255), 0.35), -1)
        cv2.circle(scene, center, 9, color, -1)
        cv2.circle(scene, center, 9, (255, 255, 255), 1)


def _draw_impacts(scene, impact_effects):
    for impact in impact_effects:
        progress = 1.0 - (impact["ttl"] / impact["max_ttl"])
        radius = int(10 + progress * 26)
        thickness = max(1, 3 - int(progress * 2))
        glow_color = _mix_color(impact["color"], (255, 255, 255), 0.25)
        center = (int(impact["x"]), int(impact["y"]))
        cv2.circle(scene, center, radius, glow_color, thickness)
        cv2.circle(scene, center, max(2, radius // 3), impact["color"], -1)


def _draw_scene(base_shape, left_target, right_gesture, enemies, projectiles, enemy_projectiles, feedback, mage_sprites, player_action_state, frame_index, heart_sprite, background_scene, legend_sprite, player_state):
    if background_scene is not None and background_scene.ndim == 3:
        scene = background_scene.copy()
    else:
        scene = np.zeros(base_shape, dtype=np.uint8)
    h, w, _ = scene.shape

    if background_scene is None:
        cv2.rectangle(scene, (0, 0), (w, h), (15, 15, 25), -1)

    cv2.line(scene, (0, 170), (w, 170), (70, 70, 90), 1)
    _draw_enemies(scene, enemies, mage_sprites, frame_index, heart_sprite)

    target_zone = left_target["zone"] if left_target else "2/3"
    active_element = right_gesture["element"] if right_gesture and right_gesture["element"] else None
    shield_active = right_gesture and right_gesture["mode"] == "DEFESA"

    player_center = (w // 2, h - 110)
    if shield_active:
        _draw_shield_dome(scene, player_center)
    _draw_player_sprite(scene, player_center, mage_sprites, active_element, player_action_state, frame_index)
    _draw_aim(scene, player_center, target_zone)
    _draw_projectiles(scene, projectiles)
    _draw_projectiles(scene, enemy_projectiles)
    _draw_player_lives(scene, player_state["lives"], player_state["max_lives"], heart_sprite)
    if legend_sprite is not None:
        legend_x = scene.shape[1] - legend_sprite.shape[1] // 2
        legend_y = scene.shape[0]
        _overlay_prepared_sprite(scene, legend_sprite, legend_x, legend_y)

    _draw_feedback_toast(scene, feedback)

    return scene


def _player_animation_state(right_gesture, player_action_timer):
    if player_action_timer > 0:
        return "attack"
    if right_gesture and right_gesture.get("element"):
        return "walk"
    return "idle"


def _target_index_from_zone(zone, enemy_count):
    zone_to_index = {
        "1/3": 0,
        "2/3": 1,
        "3/3": 2,
    }
    index = zone_to_index.get(zone, 1)
    if enemy_count <= 0:
        return None
    return max(0, min(index, enemy_count - 1))


def main():
    cap = open_camera()
    if cap is None:
        return

    cv2.namedWindow("Jogo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Jogo", GAME_VIEW_SIZE[0], GAME_VIEW_SIZE[1])
    cv2.namedWindow("Webcam", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Webcam", WEBCAM_VIEW_SIZE[0], WEBCAM_VIEW_SIZE[1])

    raw_mage_sprites = _load_mage_sprites()
    raw_player_sprite = _load_player_sprite()
    prepared_player_sprite = _resize_sprite_to_height(raw_player_sprite, 96)
    mage_sprites = {
        None: {
            "idle": [prepared_player_sprite] if prepared_player_sprite is not None else [],
            "walk": [prepared_player_sprite] if prepared_player_sprite is not None else [],
            "attack": [prepared_player_sprite] if prepared_player_sprite is not None else [],
            "hurt": [prepared_player_sprite] if prepared_player_sprite is not None else [],
            "die": [prepared_player_sprite] if prepared_player_sprite is not None else [],
        },
        "FOGO": _prepare_sprite_library({"FOGO": raw_mage_sprites["FOGO"]}, 80)["FOGO"],
        "GELO": _prepare_sprite_library({"GELO": raw_mage_sprites["GELO"]}, 80)["GELO"],
        "ELETRICO": _prepare_sprite_library({"ELETRICO": raw_mage_sprites["ELETRICO"]}, 80)["ELETRICO"],
    }

    heart_sprite = _resize_sprite_to_height(_load_heart_sprite(), 12)
    legend_sprite = _resize_sprite_to_height(_load_legend_sprite(), 120)
    background_sprite = _load_background_sprite()
    high_score = _load_high_score()
    ui_state = {
        "started": False,
        "start_requested": False,
        "game_over": False,
        "restart_requested": False,
        "start_button_rect": None,
        "restart_button_rect": None,
    }
    background_scene = None
    background_scene_size = None
    enemies = None
    projectiles = []
    enemy_projectiles = []
    impact_effects = []
    feedback = None
    feedback_timer = 0
    cast_armed = True
    player_action_timer = 0
    frame_index = 0
    player_state = {
        "lives": 5,
        "max_lives": 5,
        "score": 0,
        "high_score": high_score,
        "defeated_enemies": 0,
        "difficulty": 1,
    }

    def on_game_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if not ui_state.get("started"):
            button_rect = ui_state.get("start_button_rect")
            if button_rect and _point_in_rect(x, y, button_rect):
                ui_state["start_requested"] = True
            return
        if ui_state.get("game_over"):
            button_rect = ui_state.get("restart_button_rect")
            if button_rect and _point_in_rect(x, y, button_rect):
                ui_state["restart_requested"] = True

    cv2.setMouseCallback("Jogo", on_game_mouse)

    try:
        with create_hand_landmarker() as landmarker:
            while True:
                frame = capture_frame(cap)
                if frame is None:
                    print("Erro: nao foi possivel capturar o frame.")
                    break

                current_scene_size = (frame.shape[1], frame.shape[0])
                if background_scene is None or background_scene_size != current_scene_size:
                    background_scene = _prepare_background(background_sprite, current_scene_size)
                    background_scene_size = current_scene_size

                mp_image = frame_to_mp_image(frame)
                detection_result = landmarker.detect(mp_image)

                left_lines = []
                right_lines = []
                left_target = None
                right_gesture = None
                left_hand_landmarks = None
                right_hand_landmarks = None

                if detection_result.hand_landmarks:
                    draw_landmarks_on_frame(frame, detection_result)

                    hands_by_side = {}
                    for hand_landmarks, handedness in zip(
                        detection_result.hand_landmarks, detection_result.handedness
                    ):
                        raw_side = handedness[0].category_name
                        display_side = get_hand_side(raw_side)
                        hands_by_side[display_side] = hand_landmarks

                    left_hand_landmarks = hands_by_side.get("Esquerda")
                    right_hand_landmarks = hands_by_side.get("Direita")

                if enemies is None:
                    enemies = _create_enemies(frame.shape[1], player_state["difficulty"])

                player_center = (frame.shape[1] // 2, frame.shape[0] - 110)
                if ui_state["start_requested"]:
                    enemies, projectiles, enemy_projectiles, impact_effects, feedback, feedback_timer, cast_armed, player_action_timer = _reset_match_state(
                        player_state, frame.shape[1]
                    )
                    ui_state["start_requested"] = False
                    ui_state["started"] = True
                    ui_state["game_over"] = False

                if ui_state["restart_requested"]:
                    enemies, projectiles, enemy_projectiles, impact_effects, feedback, feedback_timer, cast_armed, player_action_timer = _reset_match_state(
                        player_state, frame.shape[1]
                    )
                    ui_state["restart_requested"] = False
                    ui_state["game_over"] = False
                    ui_state["started"] = True

                if ui_state["started"] and not ui_state["game_over"]:
                    if left_hand_landmarks is not None:
                        left_target = classify_left_hand_target(left_hand_landmarks)
                        left_lines.append(left_target["action"])
                        left_lines.append(f"Zona: {left_target['zone']}")
                        left_lines.append(f"Angulo: {left_target['angle']:.1f} deg")
                    else:
                        left_lines.append("Mostre a mao esquerda")

                    if right_hand_landmarks is not None:
                        right_gesture = classify_right_hand_gesture(right_hand_landmarks)
                        right_lines.append(right_gesture["action"])
                        if right_gesture["mode"] == "DEFESA":
                            right_lines.append("Escudo ativo")
                        elif right_gesture["element"]:
                            right_lines.append(f"Combo: {right_gesture['combo']}")
                            right_lines.append(f"Elemento: {right_gesture['element']}")
                        else:
                            right_lines.append("Sem combo")
                    else:
                        right_lines.append("Mostre a mao direita")

                    current_element = right_gesture["element"] if right_gesture else None
                    current_mode = right_gesture["mode"] if right_gesture else "NEUTRO"

                    if current_mode == "NEUTRO":
                        cast_armed = True
                    elif current_element and cast_armed:
                        target_zone = left_target["zone"] if left_target else "2/3"
                        target_index = _target_index_from_zone(target_zone, len(enemies))
                        projectile = _spawn_projectile(
                            player_center,
                            current_element,
                            target_index,
                            enemies,
                        )
                        if projectile is not None:
                            projectiles.append(projectile)
                            cast_armed = False
                            player_action_timer = 10

                    projectiles, shot_feedback = _update_projectiles(projectiles, enemies, impact_effects, player_state)
                    player_state["difficulty"] = _difficulty_from_defeats(player_state["defeated_enemies"])
                    if player_state["score"] > player_state["high_score"]:
                        player_state["high_score"] = player_state["score"]
                        _save_high_score(player_state["high_score"])
                    _tick_enemy_flashes(enemies)
                    _update_enemy_respawns(enemies, player_state["difficulty"])
                    _update_enemy_attacks(enemies, enemy_projectiles, player_center, player_state["difficulty"])
                    enemy_projectiles, enemy_feedback = _update_enemy_projectiles(
                        enemy_projectiles,
                        player_center,
                        player_state,
                        impact_effects,
                        shield_active=bool(right_gesture and right_gesture.get("mode") == "DEFESA"),
                    )
                    impact_effects = _update_impact_effects(impact_effects)

                    if enemy_feedback is not None:
                        feedback = enemy_feedback
                        feedback_timer = 24
                    elif shot_feedback is not None:
                        feedback = shot_feedback
                        feedback_timer = 30
                    elif feedback_timer > 0:
                        feedback_timer -= 1
                        if feedback_timer == 0:
                            feedback = None

                    if player_action_timer > 0:
                        player_action_timer -= 1

                    if player_state["lives"] <= 0:
                        ui_state["game_over"] = True
                        ui_state["restart_button_rect"] = None
                        feedback = None
                        feedback_timer = 0
                        player_action_timer = 0
                        cast_armed = True
                elif not ui_state["started"]:
                    left_lines.append("Jogo ainda nao iniciado")
                    right_lines.append("Clique em INICIAR")
                else:
                    left_lines.append("Jogo encerrado")
                    right_lines.append("Clique para reiniciar")

                action_state = _player_animation_state(right_gesture, player_action_timer)
                visual_frame = _draw_scene(
                    frame.shape,
                    left_target,
                    right_gesture,
                    enemies,
                    projectiles,
                    enemy_projectiles,
                    feedback,
                    mage_sprites,
                    action_state,
                    frame_index,
                    heart_sprite,
                    background_scene,
                    legend_sprite,
                    player_state,
                )
                _draw_impacts(visual_frame, impact_effects)
                _draw_game_hud(visual_frame, player_state)
                visual_frame = cv2.resize(visual_frame, GAME_VIEW_SIZE, interpolation=cv2.INTER_LINEAR)
                if not ui_state["started"]:
                    ui_state["start_button_rect"] = _draw_start_screen(visual_frame)
                elif ui_state["game_over"]:
                    ui_state["restart_button_rect"] = _draw_game_over_screen(visual_frame, player_state)
                webcam_view = cv2.resize(frame, WEBCAM_VIEW_SIZE, interpolation=cv2.INTER_LINEAR)
                _draw_panel(webcam_view, 12, "MAO ESQUERDA", left_lines, align="left")
                _draw_panel(webcam_view, 12, "MAO DIREITA", right_lines, align="right")
                cv2.putText(
                    webcam_view,
                    "ESC para sair",
                    (12, webcam_view.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                )
                # Webcam HUD removida do jogo principal para manter a tela mais limpa.
                cv2.imshow("Webcam", webcam_view)
                cv2.imshow("Jogo", visual_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break

                frame_index += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
