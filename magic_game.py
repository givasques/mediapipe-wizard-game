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
MAGE_FOLDER_BY_ELEMENT = {
    "FOGO": "wizard_fire",
    "AGUA": "wizard_ice",
    "TERRA": "wizard",
    None: "wizard",
}
MAGE_STATE_PREFIX = {
    "idle": "1_IDLE",
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
    sprite = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
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


def _pick_sprite_frame(sprite_library, element, state, frame_index):
    element_key = element if element in sprite_library else None
    state_frames = sprite_library.get(element_key, {}).get(state, [])
    if not state_frames:
        state_frames = sprite_library.get(element_key, {}).get("idle", [])
    if not state_frames:
        return None
    return state_frames[frame_index % len(state_frames)]


def _overlay_sprite(scene, sprite, anchor_x, anchor_y, target_height=190):
    if sprite is None:
        return

    sprite_h, sprite_w = sprite.shape[:2]
    if sprite_h <= 0 or sprite_w <= 0:
        return

    scale = target_height / float(sprite_h)
    target_w = max(1, int(sprite_w * scale))
    resized = cv2.resize(sprite, (target_w, target_height), interpolation=cv2.INTER_AREA)

    x1 = int(anchor_x - target_w // 2)
    y2 = int(anchor_y)
    y1 = int(y2 - target_height)

    scene_h, scene_w = scene.shape[:2]
    sx1 = max(0, x1)
    sy1 = max(0, y1)
    sx2 = min(scene_w, x1 + target_w)
    sy2 = min(scene_h, y1 + target_height)
    if sx1 >= sx2 or sy1 >= sy2:
        return

    rx1 = sx1 - x1
    ry1 = sy1 - y1
    rx2 = rx1 + (sx2 - sx1)
    ry2 = ry1 + (sy2 - sy1)

    sprite_roi = resized[ry1:ry2, rx1:rx2]
    bgr = sprite_roi[:, :, :3].astype(np.float32)
    alpha = sprite_roi[:, :, 3:4].astype(np.float32) / 255.0

    scene_roi = scene[sy1:sy2, sx1:sx2].astype(np.float32)
    blended = scene_roi * (1.0 - alpha) + bgr * alpha
    scene[sy1:sy2, sx1:sx2] = blended.astype(np.uint8)


def _element_color(element):
    return {
        "FOGO": (40, 80, 255),
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
        "GELO": (180, 240, 255),
        "ELETRICO": (255, 245, 120),
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
            {"element": "GELO", "color": (255, 180, 80)},
            {"element": "ELETRICO", "color": (0, 230, 255)},
        ]
    ).copy()


def _build_enemy(slot_x, target_y, start_at_top=False):
    template = _random_enemy_template()
    y = -35 if start_at_top else target_y
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
        "drop_speed": 1.6 + random.random() * 1.2,
        "arrived": not start_at_top,
    }


def _create_enemies(scene_width):
    positions = _enemy_positions(scene_width)
    return [_build_enemy(slot_x, 90, start_at_top=False) for slot_x in positions]


def _draw_enemies(scene, enemies):
    for enemy in enemies:
        color = _enemy_display_color(enemy)
        outline_color = (255, 255, 255)
        if enemy.get("hit_flash", 0) > 0 and enemy.get("hit_color") is not None:
            color = enemy["hit_color"]
            outline_color = enemy["hit_color"]
        center = (int(enemy["x"]), int(enemy["y"]))
        cv2.circle(scene, center, 30, (25, 25, 35), -1)
        cv2.circle(scene, center, 28, color, -1)
        cv2.circle(scene, center, 28, outline_color, 2)
        label = enemy["element"] if enemy["alive"] else f"{enemy['element']} X"
        cv2.putText(scene, label, (center[0] - 40, center[1] + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 2)

        life_y = enemy["y"] + 66
        life_w = 10
        life_h = 6
        life_gap = 6
        life_color_alive = (80, 220, 80)
        life_color_dead = (70, 70, 70)
        for idx in range(enemy["max_lives"]):
            x1 = int(center[0] - 12 + idx * (life_w + life_gap))
            y1 = int(life_y)
            x2 = x1 + life_w
            y2 = y1 + life_h
            if idx < enemy["lives"]:
                cv2.rectangle(scene, (x1, y1), (x2, y2), life_color_alive, -1)
            else:
                cv2.rectangle(scene, (x1, y1), (x2, y2), life_color_dead, -1)


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


def _update_projectiles(projectiles, enemies, impact_effects):
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
                if target["lives"] <= 0:
                    target["alive"] = False
                    target["arrived"] = False
                    target["spawn_delay"] = 20
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


def _tick_enemy_flashes(enemies):
    for enemy in enemies:
        if enemy.get("hit_flash", 0) > 0:
            enemy["hit_flash"] -= 1
            if enemy["hit_flash"] <= 0:
                enemy["hit_color"] = None


def _update_enemy_respawns(enemies):
    for enemy in enemies:
        if enemy.get("alive") and not enemy.get("arrived", True):
            enemy["y"] = min(enemy["target_y"], enemy["y"] + enemy.get("drop_speed", 2.0))
            if enemy["y"] >= enemy["target_y"]:
                enemy["y"] = enemy["target_y"]
                enemy["arrived"] = True
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
                enemy["y"] = -35
                enemy["drop_speed"] = 1.6 + random.random() * 1.2
                enemy["arrived"] = False
                enemy["spawn_delay"] = 0


def _update_impact_effects(impact_effects):
    remaining = []
    for impact in impact_effects:
        impact["ttl"] -= 1
        if impact["ttl"] > 0:
            remaining.append(impact)
    return remaining


def _draw_player(scene, center, active_element, shield_active):
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

    cv2.putText(scene, "MAGO", (x - 24, y + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)


def _draw_player_sprite(scene, center, mage_sprites, active_element, action_state, frame_index, shield_active):
    x, y = center
    sprite = _pick_sprite_frame(mage_sprites, active_element, action_state, frame_index)

    if sprite is None:
        _draw_player(scene, center, active_element, shield_active)
        return

    _overlay_sprite(scene, sprite, x, y + 35, target_height=200)

    if shield_active:
        cv2.ellipse(scene, (x, y + 10), (70, 70), 0, 200, 340, (0, 220, 255), 4)
        cv2.ellipse(scene, (x, y + 10), (70, 70), 0, 20, 160, (0, 220, 255), 2)


def _draw_aim(scene, player_center, target_zone):
    x, y = player_center
    targets = {
        "1/3": (int(scene.shape[1] * 0.2), 120),
        "2/3": (int(scene.shape[1] * 0.5), 120),
        "3/3": (int(scene.shape[1] * 0.8), 120),
    }
    end_point = targets.get(target_zone, targets["2/3"])
    cv2.arrowedLine(scene, (x, y - 20), end_point, (255, 255, 255), 3, tipLength=0.08)
    cv2.putText(scene, f"Mira {target_zone}", (x - 55, y - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


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


def _draw_scene(base_shape, left_target, right_gesture, enemies, projectiles, feedback, mage_sprites, player_action_state, frame_index):
    scene = np.zeros(base_shape, dtype=np.uint8)
    h, w, _ = scene.shape

    cv2.rectangle(scene, (0, 0), (w, h), (15, 15, 25), -1)
    cv2.line(scene, (0, 170), (w, 170), (70, 70, 90), 2)
    _draw_enemies(scene, enemies)

    target_zone = left_target["zone"] if left_target else "2/3"
    active_element = right_gesture["element"] if right_gesture and right_gesture["element"] else None
    shield_active = right_gesture and right_gesture["mode"] == "DEFESA"

    player_center = (w // 2, h - 110)
    _draw_player_sprite(scene, player_center, mage_sprites, active_element, player_action_state, frame_index, shield_active)
    _draw_aim(scene, player_center, target_zone)
    _draw_projectiles(scene, projectiles)

    if left_target:
        cv2.putText(scene, left_target["action"], (20, h - 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)
    if right_gesture:
        cv2.putText(scene, right_gesture["action"], (20, h - 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)
        if right_gesture.get("combo"):
            cv2.putText(scene, f"Combo: {right_gesture['combo']}", (20, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, _element_color(right_gesture.get("element")), 2)

    if feedback:
        cv2.putText(scene, feedback["text"], (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, feedback["color"], 2)

    return scene


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

    mage_sprites = _load_mage_sprites()
    enemies = None
    projectiles = []
    impact_effects = []
    feedback = None
    feedback_timer = 0
    cast_armed = True
    player_action_timer = 0
    frame_index = 0

    try:
        with create_hand_landmarker() as landmarker:
            while True:
                frame = capture_frame(cap)
                if frame is None:
                    print("Erro: nao foi possivel capturar o frame.")
                    break

                mp_image = frame_to_mp_image(frame)
                detection_result = landmarker.detect(mp_image)

                left_lines = []
                right_lines = []
                left_target = None
                right_gesture = None

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
                        enemies = _create_enemies(frame.shape[1])

                    if left_hand_landmarks is not None:
                        left_target = classify_left_hand_target(left_hand_landmarks, frame.shape[1])
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
                else:
                    left_lines.append("Mostre as maos")
                    right_lines.append("Mostre as maos")

                if enemies is None:
                    enemies = _create_enemies(frame.shape[1])

                current_element = right_gesture["element"] if right_gesture else None
                current_mode = right_gesture["mode"] if right_gesture else "NEUTRO"

                if current_mode == "NEUTRO":
                    cast_armed = True
                elif current_element and cast_armed:
                    target_zone = left_target["zone"] if left_target else "2/3"
                    target_index = _target_index_from_zone(target_zone, len(enemies))
                    projectile = _spawn_projectile(
                        (frame.shape[1] // 2, frame.shape[0] - 110),
                        current_element,
                        target_index,
                        enemies,
                    )
                    if projectile is not None:
                        projectiles.append(projectile)
                        cast_armed = False
                        player_action_timer = 10

                projectiles, shot_feedback = _update_projectiles(projectiles, enemies, impact_effects)
                _tick_enemy_flashes(enemies)
                _update_enemy_respawns(enemies)
                impact_effects = _update_impact_effects(impact_effects)

                if shot_feedback is not None:
                    feedback = shot_feedback
                    feedback_timer = 30
                elif feedback_timer > 0:
                    feedback_timer -= 1
                    if feedback_timer == 0:
                        feedback = None

                if player_action_timer > 0:
                    player_action_timer -= 1

                action_state = "attack" if player_action_timer > 0 else "idle"
                visual_frame = _draw_scene(
                    frame.shape,
                    left_target,
                    right_gesture,
                    enemies,
                    projectiles,
                    feedback,
                    mage_sprites,
                    action_state,
                    frame_index,
                )
                _draw_impacts(visual_frame, impact_effects)
                _draw_panel(frame, 20, "MAO ESQUERDA", left_lines, align="left")
                _draw_panel(frame, 20, "MAO DIREITA", right_lines, align="right")
                cv2.putText(
                    frame,
                    "ESC para sair",
                    (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow("Webcam", frame)
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
