#!/usr/bin/env python3
"""
Receptor piloto para el emisor 2x2.

Este script:
- abre la camara con OpenCV,
- intenta localizar automaticamente la pantalla transmitida,
- rectifica la region detectada,
- divide la ROI en una grilla 2x2,
- convierte los 4 niveles de gris en 1 byte,
- reconstruye el texto transmitido y lo muestra en vivo.

El enfoque es deliberadamente simple para que sirva como base del proyecto.
Para mejorar la tasa y robustez luego puedes:
- aumentar GRID_SIZE en ambos lados,
- agregar preambulo,
- agregar pilotos de brillo/color,
- agregar sincronizacion temporal real,
- sustituir el muestreo por un detector mas robusto.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover - depende del entorno
    raise SystemExit(
        "Falta OpenCV. Instala con: python3 -m pip install opencv-python"
    ) from exc


BITS_PER_CELL = 2
GRID_SIZE = 2
CELL_LEVELS = np.array([0, 85, 170, 255], dtype=np.float32)
SYNC_PREAMBLE = bytes([0xA5, 0x5A, 0x3C, 0xC3, 0x96, 0x69, 0xF0, 0x0F])


@dataclass
class DecodedFrame:
    """Resultado de decodificar una ROI rectificada."""

    byte_value: int
    symbols: Sequence[int]
    cell_levels: Sequence[float]
    confidence: float


def order_points(points: np.ndarray) -> np.ndarray:
    """Ordena 4 puntos como tl, tr, br, bl."""
    pts = np.asarray(points, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]  # top-left
    ordered[2] = pts[np.argmax(s)]  # bottom-right
    ordered[1] = pts[np.argmin(diff)]  # top-right
    ordered[3] = pts[np.argmax(diff)]  # bottom-left
    return ordered


def nearest_symbol(level: float) -> int:
    """Convierte una intensidad media al simbolo mas cercano."""
    return int(np.argmin(np.abs(CELL_LEVELS - level)))


def symbols_to_byte(symbols: Sequence[int]) -> int:
    """Reconstruye 1 byte desde 4 simbolos de 2 bits."""
    if len(symbols) != 4:
        raise ValueError("Se esperaban exactamente 4 simbolos")
    value = 0
    for shift, symbol in zip((6, 4, 2, 0), symbols):
        value |= (int(symbol) & 0b11) << shift
    return value


def detect_screen_roi(frame_bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    Intenta localizar automaticamente el cuadrilatero principal del emisor.

    La idea es simple:
    - convertir a gris,
    - umbralizar lo no negro,
    - dilatar para conectar las 4 celdas,
    - buscar el contorno mayor,
    - rectificar con homografia.

    Si no hay una deteccion confiable, retorna None y se usa ROI manual.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 12, 255, cv2.THRESH_BINARY)

    # La dilatacion une las celdas y ayuda a formar una sola region detectable.
    kernel = np.ones((13, 13), dtype=np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=2)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    h, w = gray.shape[:2]
    frame_area = float(h * w)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < frame_area * 0.01:
            continue

        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = order_points(box)

        width_a = np.linalg.norm(box[2] - box[3])
        width_b = np.linalg.norm(box[1] - box[0])
        height_a = np.linalg.norm(box[1] - box[2])
        height_b = np.linalg.norm(box[0] - box[3])
        side = int(max(width_a, width_b, height_a, height_b))
        if side < 50:
            continue

        dst = np.array(
            [[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(box, dst)
        warped = cv2.warpPerspective(frame_bgr, matrix, (side, side))
        return warped

    return None


def manual_roi(frame_bgr: np.ndarray) -> np.ndarray:
    """Permite seleccionar la ROI a mano si el detector automatico falla."""
    roi = cv2.selectROI(
        "Selecciona la pantalla y presiona ENTER",
        frame_bgr,
        fromCenter=False,
        showCrosshair=True,
    )
    cv2.destroyWindow("Selecciona la pantalla y presiona ENTER")
    x, y, w, h = roi
    if w == 0 or h == 0:
        raise RuntimeError("No se selecciono ninguna ROI")
    crop = frame_bgr[y : y + h, x : x + w]
    return crop.copy()


def decode_roi(roi_bgr: np.ndarray) -> DecodedFrame:
    """
    Decodifica una ROI rectificada a un byte.

    Se ignoran los bordes exteriores para no depender tanto del grosor del trazo
    ni de ligeros errores de perspectiva.
    """
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    side = min(h, w)
    gray = cv2.resize(gray, (side, side), interpolation=cv2.INTER_AREA)

    margin = max(4, int(side * 0.16))
    inner = gray[margin : side - margin, margin : side - margin]
    if inner.size == 0:
        raise ValueError("ROI demasiado pequena para decodificar")

    cell_h = inner.shape[0] / GRID_SIZE
    cell_w = inner.shape[1] / GRID_SIZE

    symbols = []
    levels = []
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            y0 = int(round(row * cell_h + cell_h * 0.2))
            y1 = int(round((row + 1) * cell_h - cell_h * 0.2))
            x0 = int(round(col * cell_w + cell_w * 0.2))
            x1 = int(round((col + 1) * cell_w - cell_w * 0.2))
            cell = inner[max(0, y0) : max(0, y1), max(0, x0) : max(0, x1)]
            if cell.size == 0:
                cell = inner[
                    int(row * cell_h) : int((row + 1) * cell_h),
                    int(col * cell_w) : int((col + 1) * cell_w),
                ]
            level = float(np.mean(cell))
            symbol = nearest_symbol(level)
            levels.append(level)
            symbols.append(symbol)

    byte_value = symbols_to_byte(symbols)
    confidence = float(np.mean([1.0 - abs(level - CELL_LEVELS[s]) / 255.0 for level, s in zip(levels, symbols)]))
    return DecodedFrame(
        byte_value=byte_value,
        symbols=symbols,
        cell_levels=levels,
        confidence=confidence,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Receptor piloto 2x2 con OpenCV.")
    parser.add_argument("--camera", type=int, default=0, help="Indice de la camara.")
    parser.add_argument(
        "--symbol-ms",
        type=int,
        default=100,
        help="Duracion esperada de cada simbolo del emisor. Ajusta esto si cambias la tasa.",
    )
    parser.add_argument(
        "--min-stable-frames",
        type=int,
        default=2,
        help="Numero minimo de frames iguales antes de aceptar un byte.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Archivo de salida opcional para guardar los bytes decodificados.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.symbol_ms <= 0:
        raise ValueError("--symbol-ms debe ser mayor que cero")
    if args.min_stable_frames <= 0:
        raise ValueError("--min-stable-frames debe ser mayor que cero")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la camara {args.camera}")

    decoded_bytes = bytearray()
    observed_symbols: deque[int] = deque(maxlen=len(SYNC_PREAMBLE))
    last_candidate: Optional[int] = None
    stable_count = 0
    last_roi: Optional[np.ndarray] = None
    use_manual_roi = False
    locked = False
    received_stream = bytearray()
    payload_length: Optional[int] = None
    lock_text = ""
    last_emit_time = 0.0
    sample_period = args.symbol_ms / 1000.0

    print("Presiona 'm' para seleccionar ROI manual, 'q' o ESC para salir.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("No se pudo leer un frame de la camara.")
                continue

            display = frame.copy()

            roi = None
            if not use_manual_roi:
                roi = detect_screen_roi(frame)
                if roi is not None:
                    last_roi = roi

            if roi is None and last_roi is not None:
                roi = last_roi

            if roi is not None:
                try:
                    decoded = decode_roi(roi)
                    now = time.monotonic()

                    # Filtro de estabilidad: un byte solo cuenta como observacion
                    # cuando se ha repetido suficientes frames consecutivos.
                    if decoded.byte_value == last_candidate:
                        stable_count += 1
                    else:
                        last_candidate = decoded.byte_value
                        stable_count = 1

                    if stable_count >= args.min_stable_frames and now >= last_emit_time:
                        last_emit_time = now + (sample_period * 0.85)
                        if not locked:
                            observed_symbols.append(decoded.byte_value)
                            if bytes(observed_symbols) == SYNC_PREAMBLE:
                                locked = True
                                lock_text = "Sincronizado"
                                received_stream.clear()
                                payload_length = None
                                print("Preambulo detectado. Bloqueando receptor...")
                        else:
                            received_stream.append(decoded.byte_value)

                            if payload_length is None and len(received_stream) >= 4:
                                payload_length = int.from_bytes(received_stream[:4], byteorder="big", signed=False)
                                print(f"Longitud reportada: {payload_length} bytes")
                                if payload_length > 10_000_000:
                                    raise RuntimeError("Longitud invalida; probable desincronizacion.")

                            if payload_length is not None and len(received_stream) >= 4 + payload_length:
                                payload = bytes(received_stream[4 : 4 + payload_length])
                                decoded_bytes = bytearray(payload)
                                print("\nMensaje completo recibido:")
                                print(payload.decode("utf-8", errors="replace"))
                                if args.output:
                                    args.output.write_bytes(payload)
                                    print(f"Bytes guardados en: {args.output}")
                                break

                    x = 10
                    y = 30
                    cv2.putText(display, f"Byte: 0x{decoded.byte_value:02X}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(
                        display,
                        f"Conf: {decoded.confidence:.2f}  {lock_text}",
                        (x, y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        display,
                        f"Obs: {len(observed_symbols)}  Stream: {len(received_stream)}",
                        (x, y + 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )
                    if roi is not None and display.shape[0] >= 240 and display.shape[1] >= 240:
                        preview = cv2.resize(roi, (240, 240), interpolation=cv2.INTER_AREA)
                        display[:240, -240:] = preview
                except Exception as exc:
                    cv2.putText(
                        display,
                        f"Decodificacion fallida: {exc}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )
            else:
                cv2.putText(
                    display,
                    "Buscando pantalla... presiona 'm' para ROI manual",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

            cv2.imshow("Receptor 2x2", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("m"):
                use_manual_roi = True
                last_roi = manual_roi(frame)
                print("ROI manual seleccionada.")

    finally:
        cap.release()
        cv2.destroyAllWindows()

    if decoded_bytes:
        print("\nMensaje decodificado:")
        print(decoded_bytes.decode("utf-8", errors="replace"))
    else:
        print("No se decodifico ningun byte.")


if __name__ == "__main__":
    main()
