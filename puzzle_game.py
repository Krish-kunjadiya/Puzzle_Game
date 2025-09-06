import cv2
import mediapipe as mp
import numpy as np
import random
import math
import time
import os

# --- Constants ---
WINDOW_WIDTH, WINDOW_HEIGHT = 1280, 720
IMAGE_FOLDER = "puzzle_images"

# Game States
STATE_PLAYING = 1
STATE_LEVEL_WON = 2

# Level Configuration
LEVELS = [(2, 2), (3, 3), (4, 4), (5, 5), (6, 6)]

# Hand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# --- UI & Colors ---
COLOR_WHITE = (255, 255, 255)
COLOR_GREEN = (100, 255, 100)
COLOR_YELLOW = (50, 255, 255)
COLOR_BLACK_TRANSPARENT = (0, 0, 0, 180)

class PuzzlePiece:
    def __init__(self, img, correct_pos_index, piece_size, puzzle_area_size):
        self.img = img
        self.correct_pos_index = correct_pos_index
        self.piece_size = piece_size
        self.is_placed = False
        self.is_dragging = False
        scatter_x_min = puzzle_area_size[0] + 50
        scatter_x_max = WINDOW_WIDTH - self.piece_size[0] - 20
        scatter_y_min = 50
        scatter_y_max = WINDOW_HEIGHT - self.piece_size[1] - 20
        self.current_pos = (random.randint(scatter_x_min, scatter_x_max), random.randint(scatter_y_min, scatter_y_max))
        self.drag_offset = (0, 0)

def get_shuffled_image_list(folder):
    """Gets a shuffled list of all valid image paths from a folder."""
    try:
        images = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(images)
        return images
    except FileNotFoundError:
        return []

def setup_level(level_index, image_path):
    """Sets up the game for a specific level using a given image path."""
    grid_rows, grid_cols = LEVELS[level_index]
    
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"ERROR: Failed to load image: {image_path}")
        return None, None, None
        
    puzzle_area_width = int(WINDOW_HEIGHT * 0.8)
    puzzle_area_height = puzzle_area_width
    puzzle_area_start_pos = (60, (WINDOW_HEIGHT - puzzle_area_height) // 2)

    h, w, _ = original_image.shape
    min_dim = min(h, w)
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    square_crop = original_image[start_y:start_y+min_dim, start_x:start_x+min_dim]
    resized_image = cv2.resize(square_crop, (puzzle_area_width, puzzle_area_height))
    
    pieces = []
    piece_height = puzzle_area_height // grid_rows
    piece_width = puzzle_area_width // grid_cols

    for r in range(grid_rows):
        for c in range(grid_cols):
            piece_img = resized_image[r*piece_height:(r+1)*piece_height, c*piece_width:(c+1)*piece_width]
            pieces.append(PuzzlePiece(piece_img, (r, c), (piece_width, piece_height), (puzzle_area_width, puzzle_area_height)))
            
    level_info = {
        "grid_rows": grid_rows, "grid_cols": grid_cols,
        "puzzle_area_width": puzzle_area_width, "puzzle_area_height": puzzle_area_height,
        "start_pos": puzzle_area_start_pos
    }
    return pieces, level_info, time.time()

def draw_text_with_bg(img, text, pos, font_scale, text_color, bg_color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size
    bg_x1, bg_y1 = pos[0] - 10, pos[1] - text_h - 10
    bg_x2, bg_y2 = pos[0] + text_w + 10, pos[1] + 10
    
    overlay = img.copy()
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)
    
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.putText(img, text, pos, font, font_scale, text_color, thickness, lineType=cv2.LINE_AA)

def are_fingers_up(hand_landmarks, required_fingers, forbidden_fingers):
    """Generic finger state checker."""
    # Check if required fingers are up (tip.y < knuckle.y)
    for finger_tip_enum, mcp_joint_enum in required_fingers:
        if hand_landmarks.landmark[finger_tip_enum].y >= hand_landmarks.landmark[mcp_joint_enum].y:
            return False
    # Check if forbidden fingers are down (tip.y > knuckle.y)
    for finger_tip_enum, mcp_joint_enum in forbidden_fingers:
        if hand_landmarks.landmark[finger_tip_enum].y < hand_landmarks.landmark[mcp_joint_enum].y:
            return False
    return True

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return
    cap.set(3, WINDOW_WIDTH)
    cap.set(4, WINDOW_HEIGHT)
    
    image_path_list = get_shuffled_image_list(IMAGE_FOLDER)
    if not image_path_list:
        print(f"--- ERROR ---")
        print(f"No images found in the '{IMAGE_FOLDER}' folder.")
        print(f"Please create the folder and add at least one .jpg or .png image.")
        print(f"-------------")
        return

    current_level = 0
    game_state = STATE_PLAYING
    
    current_image_path = image_path_list.pop()
    puzzle_pieces, level_info, start_time = setup_level(current_level, current_image_path)
    
    if puzzle_pieces is None: return

    final_time_str = ""

    while True:
        success, frame = cap.read()
        if not success: break

        frame = cv2.flip(frame, 1)
        ui_overlay = np.zeros_like(frame)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        if game_state == STATE_PLAYING:
            grid_x_start, grid_y_start = level_info["start_pos"]
            # Drawing grid, handling gestures, and puzzle logic...
            # (This part is unchanged)
            for r in range(level_info["grid_rows"] + 1):
                y = grid_y_start + r * (level_info["puzzle_area_height"] // level_info["grid_rows"])
                cv2.line(ui_overlay, (grid_x_start, y), (grid_x_start + level_info["puzzle_area_width"], y), COLOR_WHITE, 1)
            for c in range(level_info["grid_cols"] + 1):
                x = grid_x_start + c * (level_info["puzzle_area_width"] // level_info["grid_cols"])
                cv2.line(ui_overlay, (x, grid_y_start), (x, grid_y_start + level_info["puzzle_area_height"]), COLOR_WHITE, 1)

            cursor_pos = (0, 0)
            pinch_active = False
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                thumb_lm = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_lm = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                cursor_pos = (int(index_lm.x * WINDOW_WIDTH), int(index_lm.y * WINDOW_HEIGHT))
                if math.dist((thumb_lm.x, thumb_lm.y), (index_lm.x, index_lm.y)) < 0.05:
                    pinch_active = True
                    cv2.circle(frame, cursor_pos, 15, COLOR_GREEN, cv2.FILLED)
            
            if pinch_active and not any(p.is_dragging for p in puzzle_pieces):
                for piece in reversed(puzzle_pieces):
                    if not piece.is_placed:
                        px, py = piece.current_pos
                        pw, ph = piece.piece_size
                        if px < cursor_pos[0] < px + pw and py < cursor_pos[1] < py + ph:
                            piece.is_dragging = True
                            piece.drag_offset = (cursor_pos[0] - px, cursor_pos[1] - py)
                            puzzle_pieces.remove(piece)
                            puzzle_pieces.append(piece)
                            break
            
            dragging_piece = next((p for p in puzzle_pieces if p.is_dragging), None)
            if dragging_piece:
                if pinch_active:
                    dragging_piece.current_pos = (cursor_pos[0] - dragging_piece.drag_offset[0], cursor_pos[1] - dragging_piece.drag_offset[1])
                else:
                    dragging_piece.is_dragging = False
                    row, col = dragging_piece.correct_pos_index
                    pw, ph = dragging_piece.piece_size
                    target_x, target_y = grid_x_start + col * pw, grid_y_start + row * ph
                    if math.dist(dragging_piece.current_pos, (target_x, target_y)) < 30:
                        dragging_piece.current_pos = (target_x, target_y)
                        dragging_piece.is_placed = True
            
            if all(p.is_placed for p in puzzle_pieces):
                game_state = STATE_LEVEL_WON
                elapsed_time = time.time() - start_time
                minutes, seconds = divmod(int(elapsed_time), 60)
                final_time_str = f"Time: {minutes:02d}:{seconds:02d}"

            for piece in puzzle_pieces:
                x, y = piece.current_pos
                h, w, _ = piece.img.shape
                if 0 <= y < WINDOW_HEIGHT - h and 0 <= x < WINDOW_WIDTH - w:
                    ui_overlay[y:y+h, x:x+w] = piece.img
            
            elapsed_time = time.time() - start_time
            minutes, seconds = divmod(int(elapsed_time), 60)
            draw_text_with_bg(frame, f"Time: {minutes:02d}:{seconds:02d}", (WINDOW_WIDTH - 230, 40), 1.2, COLOR_YELLOW, (0,0,0))
            draw_text_with_bg(frame, f"Level {current_level + 1}", (30, 40), 1.2, COLOR_YELLOW, (0,0,0))

        elif game_state == STATE_LEVEL_WON:
            panel_x, panel_y, panel_w, panel_h = 300, 150, 680, 420
            overlay = ui_overlay.copy()
            cv2.rectangle(overlay, (panel_x, panel_y), (panel_x+panel_w, panel_y+panel_h), (20,20,20), -1)
            cv2.addWeighted(overlay, 0.75, ui_overlay, 0.25, 0, ui_overlay)
            win_text = f"LEVEL {current_level + 1} COMPLETE!"
            text_size = cv2.getTextSize(win_text, cv2.FONT_HERSHEY_TRIPLEX, 2, 3)[0]
            cv2.putText(ui_overlay, win_text, ((WINDOW_WIDTH - text_size[0]) // 2, 280), cv2.FONT_HERSHEY_TRIPLEX, 2, COLOR_GREEN, 3)
            text_size = cv2.getTextSize(final_time_str, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)[0]
            cv2.putText(ui_overlay, final_time_str, ((WINDOW_WIDTH - text_size[0]) // 2, 380), cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLOR_WHITE, 2)
            
            # --- NEW: Updated Instructions ---
            if current_level < len(LEVELS) - 1:
                cv2.putText(ui_overlay, "2 Fingers: Next Level", (440, 480), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_YELLOW, 2)
                cv2.putText(ui_overlay, "Thumbs Up: Exit", (480, 530), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_YELLOW, 2)
            else:
                cv2.putText(ui_overlay, "CONGRATULATIONS!", ((WINDOW_WIDTH - 480) // 2, 480), cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLOR_GREEN, 2)
                cv2.putText(ui_overlay, "Show Thumbs Up to Exit", ((WINDOW_WIDTH - 440) // 2, 530), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_YELLOW, 2)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Define finger landmarks using enums for clarity
                fingers = {
                    "thumb": (mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.THUMB_IP),
                    "index": (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_MCP),
                    "middle": (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_MCP),
                    "ring": (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_MCP),
                    "pinky": (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_MCP)
                }
                
                # --- NEW: Thumbs Up gesture logic ---
                thumbs_up_req = [fingers["thumb"]]
                thumbs_up_forbid = [fingers["index"], fingers["middle"], fingers["ring"], fingers["pinky"]]
                if are_fingers_up(hand_landmarks, thumbs_up_req, thumbs_up_forbid):
                    break

                # Two fingers logic (unchanged)
                if current_level < len(LEVELS) - 1:
                    two_fingers_req = [fingers["index"], fingers["middle"]]
                    two_fingers_forbid = [fingers["ring"], fingers["pinky"]]
                    if are_fingers_up(hand_landmarks, two_fingers_req, two_fingers_forbid):
                        current_level += 1
                        
                        if not image_path_list:
                            print("--- All unique images used. Reshuffling. ---")
                            image_path_list = get_shuffled_image_list(IMAGE_FOLDER)
                        
                        current_image_path = image_path_list.pop()
                        puzzle_pieces, level_info, start_time = setup_level(current_level, current_image_path)
                        if puzzle_pieces is None: break
                        game_state = STATE_PLAYING
        
        final_frame = cv2.addWeighted(frame, 0.5, ui_overlay, 0.8, 0)
        cv2.imshow("Hand Controlled Puzzle Game", final_frame)

        if cv2.waitKey(5) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()