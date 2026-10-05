#!/usr/bin/env python3
"""Webcam CCTag検出デモ。

960x540でキャプチャし、毎フレーム検出結果を重畳表示する。'q'かEscで終了。

opencv-pythonが必要(cctagpyの依存ではなくカメラ入出力用): pip install opencv-python
"""

from __future__ import annotations

import sys
import time

import cv2
import numpy as np

from cctagpy.detection import cctag_detection
from cctagpy.params import Parameters

WIDTH, HEIGHT = 960, 540
N_CROWNS = 3
MIN_IDENT_PROBA = 1e-6


def draw_markers(frame_bgr: np.ndarray, markers: list) -> np.ndarray:
    # 同じidの候補が複数出ることがあるため、品質最良の1つだけ残す
    best_by_id = {}
    for m in markers:
        if m.status != 1:
            continue
        if m.id not in best_by_id or m.quality > best_by_id[m.id].quality:
            best_by_id[m.id] = m

    for m in best_by_id.values():
        e = m.rescaled_outer_ellipse  # フル画像座標系(outer_ellipseはピラミッドレベルのローカル座標)
        center = (int(e.center[0]), int(e.center[1]))
        axes = (int(e.a), int(e.b))
        angle_deg = np.degrees(e.angle)
        cv2.ellipse(frame_bgr, center, axes, angle_deg, 0, 360, (0, 255, 0), 2)
        cv2.putText(frame_bgr, str(m.id), (center[0] + 10, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    return frame_bgr


def main() -> int:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    if not cap.isOpened():
        print("could not open the webcam", file=sys.stderr)
        return 1

    params = Parameters(n_crowns=N_CROWNS, min_ident_proba=MIN_IDENT_PROBA)
    rng = np.random.default_rng(0)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            markers = cctag_detection(gray, params, rng=rng)
            elapsed = time.perf_counter() - t0

            frame = draw_markers(frame, markers)
            cv2.putText(frame, f"{elapsed * 1000:.0f} ms  {1.0 / elapsed:.1f} fps", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("cctagpy webcam demo (q/Esc to quit)", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
