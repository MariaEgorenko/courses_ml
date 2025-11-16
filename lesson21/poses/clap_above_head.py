import cv2
import mediapipe as mp
import collections
import numpy as np


mp_pose = mp.solutions.pose
PL = mp_pose.PoseLandmark

# Вспомогательные функции
def to_np(lm, w, h):
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)

def euclid(a, b):
    return float(np.linalg.norm(a - b))

def run_pose_clap_counter(
    source=0,
    smooth_window=5,          # сглаживание расстояния между кистями
    clap_thresh_rel=0.35,     # порог «близко»: доля ширины плеч
    min_visibility=0.5,       # минимальная видимость точки
    min_frames_between=6,     # дебаунс: минимум кадров между хлопками
    draw_skeleton=True
):
    cap = cv2.VideoCapture(source)
    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Video', 720, 720)
    if not cap.isOpened():
        raise RuntimeError("Не удалось открыть видео/камеру")
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        clap_count = 0
        was_apart = True          # предыдущая фаза (кисти были далеко)
        frames_since_last = 999   # дебаунс-счётчик
        q_dist = collections.deque(maxlen=smooth_window)

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            label = "No body"
            arms_up = False

            if res.pose_landmarks:
                lms = res.pose_landmarks.landmark
                # Достаём нужные точки
                pts = {}
                for name in [
                    PL.LEFT_WRIST, PL.RIGHT_WRIST,
                    PL.LEFT_SHOULDER, PL.RIGHT_SHOULDER,
                    PL.LEFT_EYE, PL.RIGHT_EYE
                ]:
                    lm = lms[name.value]
                    pts[name] = (to_np(lm, w, h), lm.visibility)

                # Проверяем видимость ключевых точек
                needed = [PL.LEFT_WRIST, PL.RIGHT_WRIST, PL.LEFT_SHOULDER, PL.RIGHT_SHOULDER, PL.LEFT_EYE, PL.RIGHT_EYE]
                if all(pts[p][1] >= min_visibility for p in needed):
                    LW, RW = pts[PL.LEFT_WRIST][0], pts[PL.RIGHT_WRIST][0]
                    LS, RS = pts[PL.LEFT_SHOULDER][0], pts[PL.RIGHT_SHOULDER][0]
                    LE, RE = pts[PL.LEFT_EYE][0], pts[PL.RIGHT_EYE][0]

                    # Геометрия (в пикселях)
                    shoulder_width = euclid(LS, RS) + 1e-6
                    wrist_dist = euclid(LW, RW)

                    # Нормированное сближение кистей
                    wrist_dist_rel = wrist_dist / shoulder_width
                    q_dist.append(wrist_dist_rel)
                    wrist_dist_smooth = float(np.mean(q_dist))

                    # Руки подняты: обе кисти выше уровня глаз
                    eye_y = min(LE[1], RE[1])
                    arms_up = (LW[1] < eye_y) and (RW[1] < eye_y)

                    # Хлопок: кисти близко И руки подняты
                    close_enough = wrist_dist_smooth < clap_thresh_rel
                    frames_since_last += 1

                    if arms_up and close_enough and was_apart and frames_since_last >= min_frames_between:
                        clap_count += 1
                        was_apart = False
                        frames_since_last = 0
                    elif not close_enough:
                        was_apart = True


                    # Текст состояния
                    state = []
                    state.append("ARMS_UP" if arms_up else "ARMS_DOWN")
                    state.append("CLOSE" if close_enough else "APART")
                    label = f"{' | '.join(state)}  dist={wrist_dist_smooth:.2f}  cnt={clap_count}"

                    # Рисуем направляющие
                    if draw_skeleton:
                        mp.solutions.drawing_utils.draw_landmarks(
                            frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style()
                        )
                        # Линия между запястьями
                        cv2.line(frame, tuple(LW.astype(int)), tuple(RW.astype(int)), (0, 255, 255), 2)
                        # Уровень глаз
                        cv2.line(frame, (0, int(eye_y)), (w, int(eye_y)), (128, 128, 128), 1, cv2.LINE_AA)

            # HUD
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(frame, (10, 10), (20 + tw, 20 + th), (0, 0, 0), -1)
            cv2.putText(frame, label, (15, 15 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 200), 2)
            # cv2.rectangle(frame, (w - 160, 10), (w - 10, 70), (0, 0, 0), -1)
            # cv2.putText(frame, f"CLAPS: {clap_count}", (w - 150, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 220, 255), 2)

            cv2.imshow("Video", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

    cap.release()
    cv2.destroyAllWindows()


run_pose_clap_counter()