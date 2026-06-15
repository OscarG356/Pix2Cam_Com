#!/usr/bin/env python3
"""
Receptor 8x4 Color - 8 Bytes por Frame.
Decodifica Negro (00), Verde (01), Rojo (10) y Blanco (11) usando HSV.
"""

import cv2
import numpy as np
import time
from pathlib import Path

CAMERA_INDEX = 0
OUTPUT_PATH = Path("mensaje_recibido.txt")
SAMPLE_INTERVAL_MS = 200
EXPECTED_PAYLOAD_BYTES = 512 # 8 bytes preámbulo + 498 texto + 6 bytes padding

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def decode_hsv_to_bits(h, s, v):
    """Clasifica los promedios HSV de una celda en 2 bits."""
    if s < 60 and v > 150:
        return [1, 1]  # Blanco
    if v < 55:
        return [0, 0]  # Negro
    if 35 <= h <= 95:
        return [0, 1]  # Verde
    else:
        return [1, 0]  # Rojo

def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened(): return
    
    synced = False
    received_payload = bytearray()
    next_sample_time = 0.0
    sync_whites, sync_blacks = 0, 0
    
    dst_pts = np.array([[0,0], [1840,0], [1840,1040], [0,1040]], dtype="float32")
    
    try:
        while True:
            ok, frame = cap.read()
            if not ok: continue
            
            display = frame.copy()
            current_time = time.monotonic()
            
            # --- 1. LOCALIZAR EL MARCO ROJO EN HSV ---
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            mask_red1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
            mask_red2 = cv2.inRange(hsv, (160, 100, 100), (179, 255, 255))
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)
            
            contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            screen_contour = None
            if contours:
                c = max(contours, key=cv2.contourArea)
                if cv2.contourArea(c) > 5000:
                    peri = cv2.arcLength(c, True)
                    approx = cv2.approxPolyDP(c, 0.05 * peri, True)
                    if len(approx) == 4:
                        screen_contour = approx

            if screen_contour is not None:
                cv2.polylines(display, [screen_contour], True, (255, 0, 255), 3)
                
                # --- 2. ENDEREZAR LA PERSPECTIVA ---
                pts = order_points(screen_contour.reshape(4, 2))
                M = cv2.getPerspectiveTransform(pts, dst_pts)
                warped = cv2.warpPerspective(frame, M, (1840, 1040))
                warped_hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
                
                # --- 3. EXTRAER LOS 64 BITS (32 CELDAS * 2 BITS) ---
                bits_read = []
                total_v = 0
                for i in range(32):
                    row = i // 8  
                    col = i % 8
                    
                    y1 = 120 + (row * 200) + 60
                    y2 = 120 + (row * 200) + 140
                    x1 = 120 + (col * 200) + 60
                    x2 = 120 + (col * 200) + 140
                    
                    cell_roi = warped_hsv[y1:y2, x1:x2]
                    mean_hsv = np.mean(cell_roi, axis=(0, 1)) # [H, S, V] promedios
                    total_v += mean_hsv[2] # Acumulamos el Valor (Brillo)
                    
                    bits = decode_hsv_to_bits(mean_hsv[0], mean_hsv[1], mean_hsv[2])
                    bits_read.extend(bits)

                # --- 4. LÓGICA DE SINCRONIZACIÓN Y DECODIFICACIÓN ---
                avg_v = total_v / 32 # Brillo promedio de la matriz
                
                if not synced:
                    if avg_v > 180: # Detectamos frame Brillante (Blanco de Sync)
                        sync_whites += 1; sync_blacks = 0
                    elif avg_v < 60: # Detectamos frame Oscuro (Negro de Sync)
                        if sync_whites >= 1: sync_blacks += 1
                    else:
                        sync_blacks = 0

                    if sync_whites >= 1 and sync_blacks >= 1:
                        synced = True
                        print("[SYNCED] ¡Enganchado a Color (8x4)!")
                        next_sample_time = current_time + (SAMPLE_INTERVAL_MS / 1000.0)
                        
                elif synced and current_time >= next_sample_time:
                    # Reconstruir 8 Bytes (64 bits)
                    frame_bytes = bytearray()
                    for b in range(8):
                        byte_val = 0
                        for bit_idx in range(8):
                            total_bit_idx = b * 8 + bit_idx
                            byte_val |= (bits_read[total_bit_idx] << (7 - bit_idx))
                        frame_bytes.append(byte_val)
                    
                    received_payload.extend(frame_bytes)
                    print(f"Recibidos {len(received_payload)} bytes...")
                    
                    if len(received_payload) >= EXPECTED_PAYLOAD_BYTES: break
                    next_sample_time += (SAMPLE_INTERVAL_MS / 1000.0)

            else:
                cv2.putText(display, "Buscando Marco Rojo...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
            cv2.imshow("Receptor 8x4 Color", display)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')): break

    finally:
        cap.release(); cv2.destroyAllWindows()
        if received_payload:
            # Quitamos los 8 bytes de preámbulo (0x55 * 8)
            clean_payload = received_payload[8:]
            msg = clean_payload.decode('utf-8', errors='ignore')
            
            print(f"\n=== MENSAJE DECODIFICADO COLOR ===\n{msg}\n==================================")
            OUTPUT_PATH.write_bytes(clean_payload)

if __name__ == "__main__":
    main()