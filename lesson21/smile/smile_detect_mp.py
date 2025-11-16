import cv2
import numpy as np
import mediapipe as mp
import collections

# FaceMesh индексы
LEFT_MOUTH, RIGHT_MOUTH = 61, 291
UPPER_LIP_CENTER, LOWER_LIP_CENTER = 13, 14
LEFT_EYE_OUT, RIGHT_EYE_OUT = 33, 263  # для нормализации

def euclid(a, b): 
    return np.linalg.norm(a - b)

def compute_features(landmarks, w, h):
    """Возвращает признаки и опорные точки в пикселях + нормализацию на межзрачковое расстояние."""
    def p(i):
        lm = landmarks[i]
        return np.array([lm.x * w, lm.y * h], dtype=np.float32)
    L, R = p(LEFT_MOUTH), p(RIGHT_MOUTH)
    U, D = p(UPPER_LIP_CENTER), p(LOWER_LIP_CENTER)
    M = (U + D) / 2.0
    EL, ER = p(LEFT_EYE_OUT), p(RIGHT_EYE_OUT)
    eye_dist = euclid(EL, ER) + 1e-6        # масштаб лица
    mw = euclid(L, R)                        # ширина рта, px
    mh = euclid(U, D) + 1e-6                 # высота рта, px
    ratio = mw / mh                          # «шире/выше» (безразмерный)
    lift_px = ((M[1] - L[1]) + (M[1] - R[1])) / 2.0   # >0 — уголки выше центра (ось Y вниз)
    lift_n = lift_px / eye_dist              # нормализованный подъём/опускание уголков
    mh_n = mh / eye_dist                    # высота рта в долях расстояния между глазами
    return {
        "ratio": ratio,      # ширина/высота рта
        "lift_n": lift_n,    # подъём уголков (норм.)
        "mh_n": mh_n,        # высота рта (норм.)
    }, {"L":L, "R":R, "U":U, "D":D, "M":M}

def classify_expression(ratio, lift_n,
                        ratio_smile_th=2.20,
                        lift_up_th=0.030,
                        lift_down_th=0.025):
    """
    Пороги:
    - lift_* в долях межзрачкового расстояния (~0.03 ≈ 3% от eye_dist).
    - ratio_smile_th: чем выше, тем «строже» определение улыбки.
    """
    if (ratio > ratio_smile_th) and (lift_n > lift_up_th):
        return "SMILE"
    elif (lift_n < -lift_down_th):
        return "SAD"
    else:
        return "NEUTRAL"

def run_video_3states(source=0, smooth_window=7,
                      ratio_smile_th=2.20, lift_up_th=0.030, lift_down_th=0.025,
                      draw_points=True):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError("Не удалось открыть видео/камеру")

    mp_face_mesh = mp.solutions.face_mesh

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as mesh:
        q_ratio = collections.deque(maxlen=smooth_window)
        q_lift  = collections.deque(maxlen=smooth_window)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = mesh.process(rgb)
            label = "No face"
            if res.multi_face_landmarks:
                face = res.multi_face_landmarks[0].landmark
                feats, pts = compute_features(face, w, h)
                q_ratio.append(feats["ratio"])
                q_lift.append(feats["lift_n"])
                ratio_s = float(np.mean(q_ratio))
                lift_s  = float(np.mean(q_lift))
                state = classify_expression(
                    ratio_s, lift_s,
                    ratio_smile_th=ratio_smile_th,
                    lift_up_th=lift_up_th,
                    lift_down_th=lift_down_th
                )
                label = f"{state} | r={ratio_s:.2f} lift={lift_s:+.3f}"
                if draw_points:
                    for key in ["L","R","U","D","M"]:
                        x, y = pts[key].astype(int)
                        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
                # Цвет рамки/текста по классу
                color = {"SMILE": (0,200,0), "SAD": (0,0,220), "NEUTRAL": (200,200,0)}[state]
                cv2.rectangle(frame, (10,10), (w-10, 60), (0,0,0), -1)
                cv2.putText(frame, label, (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.imshow("Smile/Neutral/Sad (MediaPipe)", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
    cap.release()
    cv2.destroyAllWindows()


run_video_3states()