#!/usr/bin/env python3
"""
Receptor 8x4 Color Ultra-Robustecido (PDI Avanzado)
Decodifica Negro (00), Verde (01), Rojo (10) y Blanco (11).
Incluye Apertura Morfológica, Bounding Box de emergencia y Cromaticidad Normalizada.
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

def decode_bgr_to_bits(b, g, r):
    """
    Clasifica usando Cromaticidad Normalizada y Brillo Promedio.
    Independiza el color de las variaciones extremas de luz ambiental.
    """
    total = float(b) + float(g) + float(r)
    
    # Evitar división por cero en negros rotundos
    if total == 0: 
        return [0, 0]
        
    v = total / 3.0
    
    # 1. Umbral de Negro Dinámico por Energía
    if v < 75:  
        return [0, 0] # Negro (00)
        
    # Calcular proporciones de cromaticidad normalizada
    bn = b / total
    gn = g / total
    rn = r / total
    
    # 2. Umbral de Blanco por Simetría y Equilibrio de Canales
    if v > 140 and abs(rn - gn) < 0.08 and abs(gn - bn) < 0.08:
        return [1, 1] # Blanco (11)
        
    # 3. Dominancia Cromática para Verde vs Rojo
    if gn > rn + 0.08:
        return [0, 1]  # Verde (01)
        
    return [1, 0]  # Rojo (10)

def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened(): 
        print("Error: No se pudo abrir la cámara.")
        return
    
    synced = False
    received_payload = bytearray()
    next_sample_time = 0.0
    sync_whites, sync_blacks = 0, 0
    
    # Coordenadas de destino fijas para el aplanado rectangular (1840x1040)
    dst_pts = np.array([[0,0], [1840,0], [1840,1040], [0,1040]], dtype="float32")
    
    try:
        while True:
            ok, frame = cap.read()
            if not ok: continue
            
            display = frame.copy()
            current_time = time.monotonic()
            
            # --- 1. LOCALIZAR EL MARCO AZUL EN HSV (SISTEMA ANTIRREFLEJOS) ---
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Filtro estricto para ignorar celestes o brillos blancos de la pantalla
            mask_blue = cv2.inRange(hsv, (100, 130, 70), (135, 255, 255))
            
            # Operaciones morfológicas para destruir ruido interno y rellenar cortes en el marco azul
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
            mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            screen_contour = None
            if contours:
                c = max(contours, key=cv2.contourArea)
                if cv2.contourArea(c) > 8000: # Exigencia de tamaño mínimo
                    peri = cv2.arcLength(c, True)
                    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                    
                    if len(approx) == 4:
                        screen_contour = approx
                    else:
                        # CAÍDA DE SEGURIDAD: Si el contorno se rompe o deforma,
                        # forzamos una caja rectangular ideal de 4 puntos envolvente.
                        rect = cv2.minAreaRect(c)
                        box = cv2.boxPoints(rect)
                        screen_contour = np.int32(box).reshape(4, 1, 2)

            if screen_contour is not None:
                # Dibujar tracking en magenta en la pantalla principal
                cv2.polylines(display, [screen_contour], True, (255, 0, 255), 3)
                
                # --- 2. ENDEREZAR LA PERSPECTIVA ---
                pts = order_points(screen_contour.reshape(4, 2))
                M = cv2.getPerspectiveTransform(pts, dst_pts)
                warped = cv2.warpPerspective(frame, M, (1840, 1040))
                
                # Filtrado Gaussiano para unificar el color y matar grano digital de la cámara
                warped_filtered = cv2.GaussianBlur(warped, (11, 11), 0)
                
                # --- 3. EXTRAER LOS 64 BITS (32 CELDAS * 2 BITS) ---
                bits_read = []
                total_v = 0
                for i in range(32):
                    row = i // 8  
                    col = i % 8
                    
                    # Encontrar centro estricto de la celda
                    center_y = 120 + (row * 200) + 100
                    center_x = 120 + (col * 200) + 100
                    
                    # Ultra-ROI central de 30x30 píxeles. Ignora por completo bordes cruzados.
                    y1, y2 = center_y - 15, center_y + 15
                    x1, x2 = center_x - 15, center_x + 15
                    
                    cell_roi = warped_filtered[y1:y2, x1:x2]
                    mean_bgr = np.mean(cell_roi, axis=(0, 1))
                    total_v += np.mean(mean_bgr) 
                    
                    bits = decode_bgr_to_bits(mean_bgr[0], mean_bgr[1], mean_bgr[2])
                    bits_read.extend(bits)

                    # Dibujar cuadritos de muestreo sobre la imagen sin filtrar para depuración visual
                    cv2.rectangle(warped, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Desplegar la matriz corregida. Debe verse plana y RECTANGULAR.
                cv2.imshow("Warped Matrix (Debug)", cv2.resize(warped, (640, 360)))

                # --- 4. LÓGICA DE SINCRONIZACIÓN Y DECODIFICACIÓN ---
                avg_v = total_v / 32 
                
                if not synced:
                    if avg_v > 180: # Sync Blanco detectado
                        sync_whites += 1; sync_blacks = 0
                    elif avg_v < 60: # Sync Negro detectado (Trigger)
                        if sync_whites >= 1: sync_blacks += 1
                    else:
                        sync_blacks = 0

                    if sync_whites >= 1 and sync_blacks >= 1:
                        synced = True
                        print("[SYNCED] ¡Enganchado a Color (8x4) con PDI!")
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
            # Eliminación de preámbulo (8 bytes de 0x55)
            clean_payload = received_payload[8:]
            msg = clean_payload.decode('utf-8', errors='ignore')
            
            print(f"\n=== MENSAJE DECODIFICADO COLOR ===\n{msg}\n==================================")
            OUTPUT_PATH.write_bytes(clean_payload)

if __name__ == "__main__":
    main()