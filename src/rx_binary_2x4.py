#!/usr/bin/env python3
"""
Receptor binario 2x4 para Pix2Cam_Com.

Detecta grilla 2x4 binaria (negro/blanco) en video en vivo.
Demodula cada celda como 1 bit y reconstruye bytes.
"""

from __future__ import annotations

import argparse
import cv2
import numpy as np
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


GRID_ROWS = 2
GRID_COLS = 4
BITS_PER_FRAME = GRID_ROWS * GRID_COLS  # 8

SYNC_PREAMBLE = bytes([0xA5, 0x5A, 0x3C, 0xC3, 0x96, 0x69, 0xF0, 0x0F])


@dataclass
class DecodedFrame:
    byte_value: int
    confidence: float


def order_points(pts: np.ndarray) -> np.ndarray:
    """Ordena puntos en orden TL, TR, BR, BL."""
    rect = np.zeros((4, 2), dtype='float32')
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def detect_grid_roi(frame_bgr: np.ndarray, debug: bool = False) -> Optional[tuple[np.ndarray, int]]:
    """
    Detecta grilla binaria usando Canny edges.
    Retorna: (roi_warped, side_size) o None
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny edge detection
    edges = cv2.Canny(blurred, 100, 200)
    
    if debug:
        print(f"[DEBUG] Gray range: {gray.min()}-{gray.max()}, mean={gray.mean():.1f}")
        print(f"[DEBUG] Edge pixels: {np.count_nonzero(edges)}")
    
    # Dilate para conectar fragmentos
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    edges = cv2.dilate(edges, kernel, iterations=2)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if debug:
        print(f"[DEBUG] Contours: {len(contours)}")
    
    if not contours:
        return None
    
    h, w = gray.shape[:2]
    frame_area = h * w
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < frame_area * 0.005:
            continue
        
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = order_points(box)
        
        # Calcular lado
        w_a = np.linalg.norm(box[2] - box[3])
        w_b = np.linalg.norm(box[1] - box[0])
        h_a = np.linalg.norm(box[1] - box[2])
        h_b = np.linalg.norm(box[0] - box[3])
        side = int(max(w_a, w_b, h_a, h_b))
        
        if side < 40:
            continue
        
        # Warp a cuadrado
        dst = np.array(
            [[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(box, dst)
        warped = cv2.warpPerspective(frame_bgr, matrix, (side, side))
        
        if debug:
            print(f"[DEBUG] ROI detected: {side}x{side}")
        
        return (warped, side)
    
    return None


def decode_roi(roi: np.ndarray) -> DecodedFrame:
    """
    Decodifica grilla 2x4 binaria.
    Cada celda es negro (0) o blanco (1).
    """
    h, w = roi.shape[:2]
    
    # Dividir en grilla 2x4
    cell_h = h // GRID_ROWS
    cell_w = w // GRID_COLS
    
    bits = []
    confidence_sum = 0.0
    
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            # Extraer celda
            y0 = row * cell_h
            y1 = (row + 1) * cell_h
            x0 = col * cell_w
            x1 = (col + 1) * cell_w
            
            cell = roi[y0:y1, x0:x1]
            
            # Convertir a escala de grises
            gray_cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
            mean_intensity = gray_cell.mean()
            
            # Threshold en 127 (punto medio)
            bit = 1 if mean_intensity > 127 else 0
            bits.append(bit)
            
            # Confianza: qué tan lejos está de 127
            confidence = 1.0 - abs(mean_intensity - 127) / 127.0
            confidence_sum += confidence
    
    # Convertir bits a byte (MSB primero)
    byte_val = 0
    for i, bit in enumerate(bits):
        byte_val |= (bit << (7 - i))
    
    avg_confidence = confidence_sum / len(bits)
    
    return DecodedFrame(byte_value=byte_val, confidence=avg_confidence)


def main():
    parser = argparse.ArgumentParser(description="Receptor binario 2x4")
    parser.add_argument("--camera", type=int, default=0, help="Índice de cámara")
    parser.add_argument("--symbol-ms", type=int, default=100, help="Duración esperada del símbolo")
    parser.add_argument("--debug", action="store_true", help="Modo debug")
    parser.add_argument("--output", type=Path, help="Archivo de salida")
    
    args = parser.parse_args()
    
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"No se pudo abrir cámara {args.camera}")
        return
    
    # Reducir exposición
    cap.set(cv2.CAP_PROP_EXPOSURE, -15)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    decoded_bytes = bytearray()
    observed_symbols = deque(maxlen=len(SYNC_PREAMBLE))
    last_byte: Optional[int] = None
    stable_count = 0
    last_roi: Optional[np.ndarray] = None
    last_roi_size: Optional[int] = None
    locked = False
    received_stream = bytearray()
    payload_length: Optional[int] = None
    last_emit_time = 0.0
    sample_period = args.symbol_ms / 1000.0
    
    print("Receptor binario 2x4")
    print("Presiona 'd' para debug, 'm' para ROI manual, 'q' para salir")
    
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            
            display = frame.copy()
            
            # Detectar grilla
            result = detect_grid_roi(frame, debug=args.debug)
            roi = None
            
            if result is not None:
                detected_roi, roi_size = result
                
                # Filtro de tamaño similar al anterior
                if last_roi_size is None:
                    last_roi = detected_roi
                    last_roi_size = roi_size
                    roi = detected_roi
                else:
                    ratio = roi_size / last_roi_size
                    if 0.85 <= ratio <= 1.15:
                        last_roi = detected_roi
                        last_roi_size = roi_size
                        roi = detected_roi
                    else:
                        roi = last_roi
            else:
                roi = last_roi
            
            # Decodificar
            if roi is not None:
                try:
                    decoded = decode_roi(roi)
                    now = time.monotonic()
                    
                    # Estabilidad temporal
                    if decoded.byte_value == last_byte:
                        stable_count += 1
                        if stable_count >= 2 and now >= last_emit_time:
                            if not locked:
                                # Buscar preamble
                                observed_symbols.append(decoded.byte_value)
                                if len(observed_symbols) == len(SYNC_PREAMBLE):
                                    if bytes(observed_symbols) == SYNC_PREAMBLE:
                                        locked = True
                                        print("\n[LOCKED] Preamble detectado")
                            else:
                                # Leer payload
                                if payload_length is None:
                                    # Leer 4 bytes de longitud
                                    if len(received_stream) < 4:
                                        received_stream.append(decoded.byte_value)
                                    else:
                                        payload_length = int.from_bytes(received_stream, 'little')
                                        print(f"[PAYLOAD] Esperando {payload_length} bytes")
                                        received_stream.clear()
                                else:
                                    # Leer payload
                                    if len(received_stream) < payload_length:
                                        received_stream.append(decoded.byte_value)
                                        if len(received_stream) == payload_length:
                                            decoded_bytes.extend(received_stream)
                                            print(f"\n[COMPLETE] Mensaje recibido ({len(decoded_bytes)} bytes)")
                                            locked = False
                                            payload_length = None
                            
                            last_emit_time = now + sample_period * 0.85
                    else:
                        last_byte = decoded.byte_value
                        stable_count = 0
                    
                    # Mostrar en pantalla
                    cv2.imshow("Receptor Binario 2x4", display)
                
                except Exception as e:
                    print(f"Error decodificando: {e}")
            
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        if decoded_bytes:
            msg = decoded_bytes.decode('utf-8', errors='replace')
            print(f"\n=== MENSAJE ===\n{msg}\n")
            if args.output:
                args.output.write_bytes(decoded_bytes)
                print(f"Guardado en: {args.output}")


if __name__ == "__main__":
    main()
