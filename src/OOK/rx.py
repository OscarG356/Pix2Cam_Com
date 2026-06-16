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
SAMPLE_INTERVAL_MS = 75
EXPECTED_PAYLOAD_BYTES = 502

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
    
    # 1. Intentar forzar el autoenfoque de la cámara al arrancar
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    
    # =================================================================
    # --- FASE DE BARRIDO, ECUALIZACIÓN Y ESTABILIZACIÓN (MARCO ROJO) ---
    # =================================================================
    print("[SISTEMA] Iniciando barrido y estabilización de cámara...")
    frames_estables = 0
    REQUIRED_STABLE_FRAMES = 30 # ~1 segundo viendo el marco sin perderlo
    
    while frames_estables < REQUIRED_STABLE_FRAMES:
        ok, frame = cap.read()
        if not ok: continue
        
        display = frame.copy()
        
        # Convertimos a HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # ECUALIZACIÓN POR SOFTWARE (CLAHE)
        # Empareja el brillo de la pantalla para que el rojo resalte sin "lavarse"
        h, s, v = cv2.split(hsv)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        v = clahe.apply(v)
        hsv_eq = cv2.merge((h, s, v))
        
        # Buscamos el ROJO en la imagen ecualizada (Doble máscara)
        mask_red1 = cv2.inRange(hsv_eq, (0, 100, 100), (10, 255, 255))
        mask_red2 = cv2.inRange(hsv_eq, (160, 100, 100), (179, 255, 255))
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        
        contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        found_good_frame = False
        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 5000:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.05 * peri, True)
                if len(approx) == 4:
                    # Dibujamos un marco verde para indicar que lo estamos leyendo bien
                    cv2.polylines(display, [approx], True, (0, 255, 0), 3) 
                    found_good_frame = True

        if found_good_frame:
            frames_estables += 1
            cv2.putText(display, f"Enfocando... {frames_estables}/{REQUIRED_STABLE_FRAMES}", 
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            frames_estables = 0 # Si parpadea o se pierde, reinicia
            cv2.putText(display, "Buscando marco ROJO para ajustar...", 
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Fase de Calibracion", display)
        if cv2.waitKey(1) & 0xFF in (27, ord('q')):
            cap.release(); cv2.destroyAllWindows(); return

    print("[SISTEMA] ¡Cámara estabilizada y marco enganchado!")
    cv2.destroyWindow("Fase de Calibracion")
    # =================================================================
    
    # --- AQUÍ INICIA TU LÓGICA DE RECEPCIÓN ORIGINAL ---
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
            
            # Opcional: Podrías aplicar CLAHE aquí también si ves que 
            # al pasar a la fase de lectura se vuelve a perder por la luz.
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
                warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                
                # --- 3. EXTRAER LOS 32 BITS ---
                bits_read = []
                for i in range(32):
                    row = i // 8  
                    col = i % 8
                    
                    y1 = 120 + (row * 200) + 50
                    y2 = 120 + (row * 200) + 150
                    x1 = 120 + (col * 200) + 50
                    x2 = 120 + (col * 200) + 150
                    
                    cell_roi = warped_gray[y1:y2, x1:x2]
                    mean_val = np.mean(cell_roi)
                    bits_read.append(1 if mean_val > 120 else 0)

                # --- 4. LÓGICA DE SINCRONIZACIÓN Y DECODIFICACIÓN ---
                sum_bits = sum(bits_read)
                
                if not synced:
                    if sum_bits >= 30: 
                        sync_whites += 1; sync_blacks = 0
                    elif sum_bits <= 2: 
                        if sync_whites >= 1: sync_blacks += 1
                    else:
                        sync_blacks = 0

                    if sync_whites >= 1 and sync_blacks >= 1:
                        synced = True
                        print("[SYNCED] 8x4 Enganchado!")
                        next_sample_time = current_time + (SAMPLE_INTERVAL_MS / 1000.0)
                        
                elif synced and current_time >= next_sample_time:
                    word32 = 0
                    for i in range(32):
                        word32 |= (bits_read[i] << (31 - i))
                    
                    byte1 = (word32 >> 24) & 0xFF
                    byte2 = (word32 >> 16) & 0xFF
                    byte3 = (word32 >> 8) & 0xFF
                    byte4 = word32 & 0xFF
                    
                    received_payload.extend([byte1, byte2, byte3, byte4])
                    print(f"Recibidos {len(received_payload)} bytes...")
                    
                    if len(received_payload) >= EXPECTED_PAYLOAD_BYTES: break
                    next_sample_time += (SAMPLE_INTERVAL_MS / 1000.0)

            else:
                cv2.putText(display, "Buscando Marco Rojo...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
            cv2.imshow("Receptor 4x4", display)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')): break

    finally:
        cap.release(); cv2.destroyAllWindows()
        if received_payload:
            clean_payload = received_payload[4:]
            msg = clean_payload.decode('utf-8', errors='ignore')
            
            print(f"\n=== MENSAJE DECODIFICADO ===\n{msg}\n============================")
            OUTPUT_PATH.write_bytes(clean_payload)

if __name__ == "__main__":
    main()