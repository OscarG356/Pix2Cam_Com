#!/usr/bin/env python3
"""
Receptor 8x4 CSK - Optimizado para Laboratorio (Luces encendidas y ángulo <= 15°)
Decodifica Negro (00), Verde (01), Rojo (10) y Blanco (11) usando espacio HSV robusto.
Versión limpia sin ventanas secundarias de depuración.
"""

import cv2
import numpy as np
import time
from pathlib import Path

CAMERA_INDEX = 0
OUTPUT_PATH = Path("mensaje_recibido.txt")
SAMPLE_INTERVAL_MS = 200
EXPECTED_PAYLOAD_BYTES = 512

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
    """
    Decodificador adaptativo basado en el modelo de color HSV.
    Ignora las variaciones de brillo (V) causadas por las luces del aula.
    """
    # 1. Clasificación por brillo y saturación para casos extremos
    if v < 65:  
        return [0, 0] # Negro (00)
        
    if s < 45 and v > 130:
        return [1, 1] # Blanco (11)

    # 2. Clasificación matemática por Matiz (Hue)
    if 35 <= h <= 95:
        return [0, 1] # Verde (01)
        
    if (0 <= h < 20) or (155 <= h <= 180):
        return [1, 0] # Rojo (10)

    if v > 120:
        return [1, 1]
    return [0, 0]

def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened(): 
        print("Error: No se pudo abrir la cámara.")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
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
            
            # --- 1. LOCALIZAR EL MARCO AZUL EN HSV ---
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask_blue = cv2.inRange(hsv, (95, 100, 60), (135, 255, 255))
            
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
            mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            screen_contour = None
            if contours:
                c = max(contours, key=cv2.contourArea)
                if cv2.contourArea(c) > 6000:
                    peri = cv2.arcLength(c, True)
                    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                    
                    if len(approx) == 4:
                        screen_contour = approx
                    else:
                        rect = cv2.minAreaRect(c)
                        box = cv2.boxPoints(rect)
                        screen_contour = np.int32(box).reshape(4, 1, 2)

            if screen_contour is not None:
                # Dibujar el contorno del marco azul detectado
                cv2.polylines(display, [screen_contour], True, (255, 0, 255), 3)
                
                # --- 2. CORRECCIÓN DE PERSPECTIVA ---
                pts = order_points(screen_contour.reshape(4, 2))
                M = cv2.getPerspectiveTransform(pts, dst_pts)
                warped = cv2.warpPerspective(frame, M, (1840, 1040))
                
                warped_blur = cv2.GaussianBlur(warped, (9, 9), 0)
                warped_hsv = cv2.cvtColor(warped_blur, cv2.COLOR_BGR2HSV)
                
                # --- 3. EXTRAER LOS 64 BITS ---
                bits_read = []
                total_v = 0
                for i in range(32):
                    row = i // 8  
                    col = i % 8
                    
                    center_y = 120 + (row * 200) + 100
                    center_x = 120 + (col * 200) + 100
                    
                    # Ventana estricta central de 20x20 píxeles
                    y1, y2 = center_y - 10, center_y + 10
                    x1, x2 = center_x - 10, center_x + 10
                    
                    cell_roi = warped_hsv[y1:y2, x1:x2]
                    mean_hsv = np.mean(cell_roi, axis=(0, 1)) 
                    total_v += mean_hsv[2] 
                    
                    bits = decode_hsv_to_bits(mean_hsv[0], mean_hsv[1], mean_hsv[2])
                    bits_read.extend(bits)

                # --- 4. LÓGICA DE SINCRONIZACIÓN ---
                avg_v = total_v / 32 
                
                if not synced:
                    if avg_v > 170: 
                        sync_whites += 1; sync_blacks = 0
                    elif avg_v < 65: 
                        if sync_whites >= 1: sync_blacks += 1
                    else:
                        sync_blacks = 0

                    if sync_whites >= 1 and sync_blacks >= 1:
                        synced = True
                        print("[CONECTADO] Calibrado para ambiente de Laboratorio.")
                        next_sample_time = current_time + (SAMPLE_INTERVAL_MS / 1000.0)
                        
                elif synced and current_time >= next_sample_time:
                    frame_bytes = bytearray()
                    for b in range(8):
                        byte_val = 0
                        for bit_idx in range(8):
                            total_bit_idx = b * 8 + bit_idx
                            byte_val |= (bits_read[total_bit_idx] << (7 - bit_idx))
                        frame_bytes.append(byte_val)
                    
                    received_payload.extend(frame_bytes)
                    print(f"Recibidos {len(received_payload)} bytes...")
                    
                    if len(received_payload) >= EXPECTED_PAYLOAD_BYTES: 
                        break
                    next_sample_time += (SAMPLE_INTERVAL_MS / 1000.0)

            else:
                cv2.putText(display, "Buscando Marco Azul...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
            cv2.imshow("Receptor 8x4 Color", display)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')): break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if received_payload:
            clean_payload = received_payload[8:]
            msg = clean_payload.decode('utf-8', errors='ignore')
            print(f"\n=== MENSAJE DECODIFICADO (HSV) ===\n{msg}\n==================================")
            OUTPUT_PATH.write_bytes(clean_payload)

if __name__ == "__main__":
    main()