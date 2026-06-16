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
SAMPLE_INTERVAL_MS = 145
EXPECTED_PAYLOAD_BYTES = 512 # 8 bytes preámbulo + 498 texto + 6 bytes padding

# Umbrales BGR para decodificación
BGR_WHITE_THR = 180
BGR_BLACK_THR = 80
BGR_DIFF_COLOR = 40

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def decode_bgr_to_bits(b, g, r):
    """Clasifica los promedios BGR de una celda en 2 bits."""
    # Blanco: Todos los canales altos
    if b > BGR_WHITE_THR and g > BGR_WHITE_THR and r > BGR_WHITE_THR:
        return [1, 1]
    # Negro: Todos los canales bajos
    if b < BGR_BLACK_THR and g < BGR_BLACK_THR and r < BGR_BLACK_THR:
        return [0, 0]
    # Verde vs Rojo (Comparación de dominancia)
    if g > r + BGR_DIFF_COLOR:
        return [0, 1]  # Verde
    return [1, 0]      # Rojo

def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened(): return
    
    # 1. Intentar forzar el autoenfoque de la cámara (Puede que tu cámara lo ignore, pero vale la pena intentar)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    
    # =================================================================
    # --- NUEVA FASE DE BARRIDO Y ESTABILIZACIÓN ---
    # =================================================================
    print("[SISTEMA] Iniciando barrido y estabilización de cámara...")
    frames_estables = 0
    REQUIRED_STABLE_FRAMES = 30 # Necesitamos ver el marco claro por ~1 segundo
    
    while frames_estables < REQUIRED_STABLE_FRAMES:
        ok, frame = cap.read()
        if not ok: continue
        
        display = frame.copy()
        
        # Convertimos a HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # ECUALIZACIÓN POR SOFTWARE (Opcional, pero ayuda si la pantalla brilla mucho)
        # Separamos los canales y aplicamos ecualización adaptativa al brillo (V)
        h, s, v = cv2.split(hsv)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        v = clahe.apply(v)
        hsv_eq = cv2.merge((h, s, v))
        
        # Buscamos el azul en la imagen ecualizada
        mask_blue = cv2.inRange(hsv_eq, (100, 100, 100), (130, 255, 255))
        contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        found_good_frame = False
        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 5000:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.05 * peri, True)
                if len(approx) == 4:
                    cv2.polylines(display, [approx], True, (0, 255, 0), 3) # Marco verde de calibración
                    found_good_frame = True

        if found_good_frame:
            frames_estables += 1
            cv2.putText(display, f"Enfocando... {frames_estables}/{REQUIRED_STABLE_FRAMES}", 
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            frames_estables = 0 # Si pierde el marco, reinicia el conteo
            cv2.putText(display, "Buscando pantalla para ajustar...", 
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Fase de Calibracion", display)
        if cv2.waitKey(1) & 0xFF in (27, ord('q')):
            cap.release(); cv2.destroyAllWindows(); return

    print("[SISTEMA] ¡Cámara estabilizada y marco enganchado!")
    cv2.destroyWindow("Fase de Calibracion")
    # =================================================================
    
    # Variables originales del sistema
    synced = False
    received_payload = bytearray()
    next_sample_time = 0.0
    sync_whites, sync_blacks = 0, 0
    
    dst_pts = np.array([[0,0], [1840,0], [1840,1040], [0,1040]], dtype="float32")
    
    try:
        while True:
            ok, frame = cap.read()
            # ... AQUÍ CONTINÚA TU BUCLE WHILE ORIGINAL ...
            # Recuerda que si el brillo sigue siendo un problema en la fase de lectura, 
            # puedes usar la misma técnica de separar HSV y aplicar `clahe` a `v` antes de leer los colores.
            if not ok: continue
            
            display = frame.copy()
            current_time = time.monotonic()
            
            # --- 1. LOCALIZAR EL MARCO AZUL EN HSV (Más robusto que RGB para tracking) ---
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # Rango para Azul
            mask_blue = cv2.inRange(hsv, (100, 100, 100), (130, 255, 255))
            contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
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
                    
                    cell_roi = warped[y1:y2, x1:x2]
                    mean_bgr = np.mean(cell_roi, axis=(0, 1)) # [B, G, R]
                    total_v += np.mean(mean_bgr) # Brillo promedio para Sync
                    
                    bits = decode_bgr_to_bits(mean_bgr[0], mean_bgr[1], mean_bgr[2])
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
                cv2.putText(display, "Buscando Marco Azul...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
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