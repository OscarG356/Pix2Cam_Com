#!/usr/bin/env python3
"""
Receptor 4x4 (Blanco y Negro) - 2 Bytes por Frame.
Usa tracking de un marco Verde Neón y umbralización de escala de grises.
"""

import cv2
import numpy as np
import time
from pathlib import Path

CAMERA_INDEX = 0
OUTPUT_PATH = Path("mensaje_recibido.txt")
SAMPLE_INTERVAL_MS = 150       
EXPECTED_PAYLOAD_BYTES = 499

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened(): return
    
    synced = False
    received_payload = bytearray()
    next_sample_time = 0.0
    sync_whites, sync_blacks = 0, 0
    
    # El nuevo marco rojo aplanado es de 1040x1040
    dst_pts = np.array([[0,0], [1040,0], [1040,1040], [0,1040]], dtype="float32")
    
    try:
        while True:
            ok, frame = cap.read()
            if not ok: continue
            
            display = frame.copy()
            current_time = time.monotonic()
            
            # --- 1. LOCALIZAR EL MARCO ROJO EN HSV ---
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Rango estricto para Rojo brillante
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
                warped = cv2.warpPerspective(frame, M, (1040, 1040)) # <-- Cambiar a 1040
                warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                
                # Convertir a escala de grises para leer el B/N
                warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                
                # --- 3. EXTRAER LOS 16 BITS ---
                bits_read = []
                for i in range(16):
                    row = i // 4
                    col = i % 4
                    
                    # Sumamos 120px de offset (40px del marco rojo + 80px del foso negro)
                    y1 = 120 + (row * 200) + 50
                    y2 = 120 + (row * 200) + 150
                    x1 = 120 + (col * 200) + 50
                    x2 = 120 + (col * 200) + 150
                    
                    cell_roi = warped_gray[y1:y2, x1:x2]
                    mean_val = np.mean(cell_roi)
                    
                    bits_read.append(1 if mean_val > 120 else 0)

                cv2.imshow("Warped B/W", warped_gray)

                # --- 4. LÓGICA DE SINCRONIZACIÓN Y DECODIFICACIÓN ---
                sum_bits = sum(bits_read)
                
                if not synced:
                    # Sync 1: Todo Blanco (16 bits en 1)
                    if sum_bits >= 15: # Permitimos 1 bit de error de tolerancia
                        sync_whites += 1; sync_blacks = 0
                    # Sync 2: Todo Negro (0 bits en 1)
                    elif sum_bits <= 1: 
                        if sync_whites >= 1: sync_blacks += 1
                    else:
                        sync_blacks = 0

                    if sync_whites >= 1 and sync_blacks >= 1:
                        synced = True
                        print("[SYNCED] 4x4 Enganchado!")
                        next_sample_time = current_time + (SAMPLE_INTERVAL_MS / 1000.0)
                        
                elif synced and current_time >= next_sample_time:
                    # Reconstruir los 16 bits en un entero
                    word16 = 0
                    for i in range(16):
                        word16 |= (bits_read[i] << (15 - i))
                    
                    # Separar en 2 bytes
                    byte1 = (word16 >> 8) & 0xFF
                    byte2 = word16 & 0xFF
                    
                    received_payload.append(byte1)
                    received_payload.append(byte2)
                    
                    print(f"Bytes {len(received_payload)}/{EXPECTED_PAYLOAD_BYTES}: [{hex(byte1)}] [{hex(byte2)}]")
                    
                    if len(received_payload) >= EXPECTED_PAYLOAD_BYTES: break
                    next_sample_time += (SAMPLE_INTERVAL_MS / 1000.0)

            else:
                cv2.putText(display, "Buscando Marco Rojo...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
            cv2.imshow("Receptor 4x4", display)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')): break

    finally:
        cap.release(); cv2.destroyAllWindows()
        if received_payload:
            # Quitamos los 2 bytes de preámbulo (0x55, 0x55)
            clean_payload = received_payload[2:]
            msg = clean_payload.decode('utf-8', errors='ignore')
            
            print(f"\n=== MENSAJE DECODIFICADO (4x4) ===\n{msg}\n==================================")
            OUTPUT_PATH.write_bytes(clean_payload)

if __name__ == "__main__":
    main()