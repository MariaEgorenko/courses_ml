import cv2
import mediapipe as mp
import numpy as np
import collections

mp_pose = mp.solutions.pose
PL = mp_pose.PoseLandmark

def to_np(lm, w, h): return np.array([lm.x * w, lm.y * h], dtype=np.float32)
def euclid(a, b): return float(np.linalg.norm(a - b))

def run_pose_clap_counter_jump(
    source=0,
    # --- хлопок (как раньше, гистерезис) ---
    smooth_window=5,
    close_th_rel=0.38,
    open_th_rel=0.60,
    min_visibility=0.5,
    min_frames_between=6,
    require_above_head=True,
    use_index_tips=True,
    # --- прыжок ---
    up_start_delta_rel=0.10,      # насколько таз поднялся от "земли", чтобы считать прыжок начатым (доля ширины плеч)
    up_vel_th_rel=0.015,          # минимальная "скорость" вверх (на кадр, в долях ширины плеч)
    ground_return_delta_rel=0.03, # считаем, что вернулись на землю, когда подъём < этого порога
    one_clap_per_jump=True,       # только 1 хлопок за прыжок
    draw_skeleton=True
):
    cap = cv2.VideoCapture(source)
    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Video', 1280, 960)
    if not cap.isOpened():
        raise RuntimeError("Не удалось открыть видео/камеру")

    with mp_pose.Pose(
        static_image_mode=False, model_complexity=1,
        enable_segmentation=False, smooth_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as pose:

        clap_count = 0
        frames_since_last = 999

        # сглаживание расстояния между руками
        q_dist = collections.deque(maxlen=smooth_window)
        prev_smooth = None
        state_close = False

        # прыжок: отслеживаем вертикаль таза
        prev_hip_y = None
        hip_ground_ema = None       # "уровень земли" (эксп. среднее)
        ema_alpha = 0.02            # скорость обновления уровня земли

        jump_state = "GROUND"       # GROUND | ASCEND | AIRBORNE | DESCEND
        clap_in_this_jump = False

        while True:
            ok, frame = cap.read()
            if not ok: break
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            hud = "No body"
            if res.pose_landmarks:
                lms = res.pose_landmarks.landmark

                names = [
                    PL.LEFT_SHOULDER, PL.RIGHT_SHOULDER,
                    PL.LEFT_EYE, PL.RIGHT_EYE,
                    PL.LEFT_HIP, PL.RIGHT_HIP,
                    PL.LEFT_WRIST, PL.RIGHT_WRIST,
                    PL.LEFT_INDEX, PL.RIGHT_INDEX
                ]
                pts = {n: (to_np(lms[n.value], w, h), lms[n.value].visibility) for n in names}

                need = [PL.LEFT_SHOULDER, PL.RIGHT_SHOULDER, PL.LEFT_EYE, PL.RIGHT_EYE, PL.LEFT_HIP, PL.RIGHT_HIP]
                if all(pts[n][1] >= min_visibility for n in need):

                    # геометрия
                    LS, RS = pts[PL.LEFT_SHOULDER][0], pts[PL.RIGHT_SHOULDER][0]
                    shoulder_width = euclid(LS, RS) + 1e-6
                    mid_shoulders_x = (LS[0] + RS[0]) / 2.0

                    hipL, hipR = pts[PL.LEFT_HIP][0], pts[PL.RIGHT_HIP][0]
                    hip_y = float((hipL[1] + hipR[1]) / 2.0)   # ниже = больше; вверх = меньше

                    # уровень "земли" (обновляем, пока мы реально на земле)
                    if hip_ground_ema is None:
                        hip_ground_ema = hip_y
                    # критерий "похоже стоим": небольшая вертикальная скорость и руки не над головой
                    # (чтобы в верхней фазе прыжка не портить уровень земли)
                    if prev_hip_y is not None:
                        vel_y = hip_y - prev_hip_y              # <0 вверх, >0 вниз
                    else:
                        vel_y = 0.0
                    prev_hip_y = hip_y

                    # нормализованные величины прыжка
                    # чем больше delta_up_rel, тем выше относительно земли мы поднялись
                    delta_up_rel = (hip_ground_ema - hip_y) / shoulder_width
                    vel_up_rel = (-vel_y) / shoulder_width     # >0 движемся вверх

                    # обновление уровня земли (если явно не прыгаем)
                    if abs(vel_up_rel) < 0.005 and delta_up_rel < 0.02:
                        hip_ground_ema = (1 - ema_alpha) * hip_ground_ema + ema_alpha * hip_y

                    # детектор прыжка (простая машина состояний)
                    if jump_state == "GROUND":
                        if delta_up_rel > up_start_delta_rel or vel_up_rel > up_vel_th_rel:
                            jump_state = "ASCEND"
                            clap_in_this_jump = False
                    elif jump_state == "ASCEND":
                        if vel_up_rel <= 0:          # достигли вершины
                            jump_state = "AIRBORNE"
                    elif jump_state == "AIRBORNE":
                        if vel_up_rel < -0.005:      # уверенно пошли вниз
                            jump_state = "DESCEND"
                    elif jump_state == "DESCEND":
                        if delta_up_rel < ground_return_delta_rel:
                            jump_state = "GROUND"

                    # точки рук: указательные или запястья
                    use_index = use_index_tips and pts[PL.LEFT_INDEX][1] >= min_visibility and pts[PL.RIGHT_INDEX][1] >= min_visibility
                    Lp = pts[PL.LEFT_INDEX][0] if use_index else pts[PL.LEFT_WRIST][0]
                    Rp = pts[PL.RIGHT_INDEX][0] if use_index else pts[PL.RIGHT_WRIST][0]

                    # расстояние между руками (норм.)
                    raw_dist = euclid(Lp, Rp)
                    dist_rel = raw_dist / shoulder_width
                    q_dist.append(dist_rel)
                    smooth = float(np.mean(q_dist))
                    deriv = 0.0 if prev_smooth is None else (smooth - prev_smooth)  # >0 расходятся
                    prev_smooth = smooth

                    # руки подняты и над головой?
                    eye_y = min(pts[PL.LEFT_EYE][0][1], pts[PL.RIGHT_EYE][0][1])
                    arms_up = (Lp[1] < eye_y) and (Rp[1] < eye_y)

                    above_ok = True
                    if require_above_head:
                        margin = 0.25 * shoulder_width
                        head_top_y = eye_y - 0.25 * shoulder_width
                        center_ok = (abs(((Lp[0]+Rp[0])/2.0) - mid_shoulders_x) < 0.6 * shoulder_width)
                        above_ok = (Lp[1] < head_top_y and Rp[1] < head_top_y and center_ok)

                    frames_since_last += 1
                    jump_active = jump_state in ("ASCEND", "AIRBORNE", "DESCEND")

                    # --- гистерезис для хлопка ---
                    if arms_up and above_ok and smooth < close_th_rel and jump_active:
                        state_close = True

                    ready_to_count = (
                        state_close and               # были "очень близко"
                        deriv > 0 and                 # минимум расстояния пройден
                        smooth > open_th_rel and      # разошлись достаточно
                        frames_since_last >= min_frames_between and
                        arms_up and above_ok and
                        jump_active and
                        (not one_clap_per_jump or not clap_in_this_jump)
                    )
                    if ready_to_count:
                        clap_count += 1
                        frames_since_last = 0
                        state_close = False
                        clap_in_this_jump = True

                    # HUD
                    state_txt = []
                    state_txt.append(f"JUMP:{jump_state}")
                    state_txt.append("ARMS_UP" if arms_up else "ARMS_DOWN")
                    state_txt.append("ABOVE_OK" if above_ok else "LOW/ASIDE")
                    state_txt.append("CLOSE" if state_close else "OPEN")
                    hud = f"{' | '.join(state_txt)}  d={smooth:.2f} d'={deriv:+.3f}  up={delta_up_rel:+.3f} v_up={vel_up_rel:+.3f}  cnt={clap_count}"

                    if draw_skeleton:
                        mp.solutions.drawing_utils.draw_landmarks(
                            frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style()
                        )
                        # рука-рука
                        cv2.line(frame, tuple(Lp.astype(int)), tuple(Rp.astype(int)), (0,255,255), 1)
                        # линия глаз
                        cv2.line(frame, (0, int(eye_y)), (w, int(eye_y)), (128,128,128), 1, cv2.LINE_AA)
                        # уровень "земли" для таза (визуально)
                        gy = int(hip_ground_ema)
                        cv2.line(frame, (0, gy), (w, gy), (80,80,80), 1, cv2.LINE_AA)

            # отрисовка HUD
            (tw, th), _ = cv2.getTextSize(hud, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            # cv2.rectangle(frame, (10, 10), (20 + tw, 20 + th), (0, 0, 0), -1)
            # cv2.putText(frame, hud, (15, 15 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1)
            # cv2.rectangle(frame, (w - 160, 10), (w - 10, 70), (0, 0, 0), -1)
            cv2.putText(frame, f"CLAPS: {clap_count}", (20 + tw, 20 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 2)

            cv2.imshow("Video", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


run_pose_clap_counter_jump()