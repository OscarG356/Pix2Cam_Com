#!/usr/bin/env python3
import cv2
import numpy as np
import time
from pathlib import Path

CAMERA_INDEX = 0
OUTPUT_PATH = Path("mensaje_recibido.txt")
SAMPLE_INTERVAL_MS = 150       
EXPECTED_PAYLOAD_BYTES = 55 
COLOR_TO_SYMBOL = {"BLACK": 0, "RED": 1, "GREEN": 2, "BLUE": 3}

def order_points(pts):
    """Ordena las 4 esquinas: Top-Left, Top-Right, Bottom-Right, Bottom-Left"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def detect_dominant_color(roi_frame: np.ndarray) -> tuple[str, float]:
    hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
    
    # 1. NEGRO: Es muy oscuro (Valor bajo), sin importar el color o la saturación
    mask_black = cv2.inRange(hsv, (0, 0, 0), (179, 255, 80))
    
    # Para los colores vivos, exigimos que brillen (V > 80) y que tengan color (S > 80)
    # 2. ROJO: En HSV el rojo está en los dos extremos (0-10 y 160-179)
    mask_red1 = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255))
    mask_red2 = cv2.inRange(hsv, (160, 80, 80), (179, 255, 255))
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    
    # 3. VERDE: Bien centrado en la zona verde (40-80)
    mask_green = cv2.inRange(hsv, (40, 80, 80), (85, 255, 255))
    
    # 4. AZUL: Bien centrado en la zona azul (100-140)
    mask_blue = cv2.inRange(hsv, (100, 80, 80), (140, 255, 255))
    
    # 5. CYAN (Solo para Sincronización): Entre verde y azul
    mask_cyan = cv2.inRange(hsv, (86, 80, 80), (99, 255, 255))
    
    area = roi_frame.shape[0] * roi_frame.shape[1]
    
    scores = {
        "BLACK": cv2.countNonZero(mask_black) / area,
        "RED": cv2.countNonZero(mask_red) / area,
        "GREEN": cv2.countNonZero(mask_green) / area,
        "BLUE": cv2.countNonZero(mask_blue) / area,
        "CYAN": cv2.countNonZero(mask_cyan) / area
    }
    
    best_color = max(scores, key=scores.get)
    return best_color, scores[best_color]

def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened(): return
    
    synced = False
    received_payload = bytearray()
    next_sample_time = 0.0
    red_count, black_count = 0, 0
    
    # Destino de la perspectiva: Un cuadrado perfecto de 400x400
    dst_pts = np.array([[0,0], [400,0], [400,400], [0,400]], dtype="float32")
    
    try:
        while True:
            ok, frame = cap.read()
            if not ok: continue
            
            display = frame.copy()
            current_time = time.monotonic()
            
            # --- 1. LOCALIZAR EL MARCO BLANCO ---
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Umbral muy alto para detectar solo el blanco brillante del marco
            _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            screen_contour = None
            if contours:
                # Tomar el contorno más grande
                c = max(contours, key=cv2.contourArea)
                if cv2.contourArea(c) > 3000:
                    peri = cv2.arcLength(c, True)
                    approx = cv2.approxPolyDP(c, 0.05 * peri, True)
                    if len(approx) == 4:
                        screen_contour = approx

            # Si encontramos la pantalla, procesamos la matriz
            if screen_contour is not None:
                cv2.polylines(display, [screen_contour], True, (0, 255, 0), 3)
                
                # --- 2. ENDEREZAR LA PERSPECTIVA ---
                pts = order_points(screen_contour.reshape(4, 2))
                M = cv2.getPerspectiveTransform(pts, dst_pts)
                warped = cv2.warpPerspective(frame, M, (400, 400))
                
                # --- 3. EXTRAER LOS 4 CUADRANTES DE LA IMAGEN PLANA ---
                # Tomamos los núcleos (evitando los bordes de cada cuadrante)
                roi_tl = warped[50:150, 50:150]
                roi_tr = warped[50:150, 250:350]
                roi_bl = warped[250:350, 50:150]
                roi_br = warped[250:350, 250:350]
                
                c_tl, s_tl = detect_dominant_color(roi_tl)
                c_tr, _ = detect_dominant_color(roi_tr)
                c_bl, _ = detect_dominant_color(roi_bl)
                c_br, _ = detect_dominant_color(roi_br)
                
                # Mostrar en una ventanita lo que la cámara "ha enderezado"
                cv2.imshow("Vista Plana (Warped)", warped)

                # --- 4. LÓGICA DE SINCRONIZACIÓN Y DECODIFICACIÓN ---
                if not synced:
                    # Ahora buscamos el cuadrante completo en CYAN
                    if all(c == "CYAN" for c in [c_tl, c_tr, c_bl, c_br]) and s_tl > 0.15:
                        red_count += 1; black_count = 0
                    elif all(c == "BLACK" for c in [c_tl, c_tr, c_bl, c_br]) and s_tl > 0.15:
                        if red_count >= 1: black_count += 1

                    if red_count >= 1 and black_count >= 1:
                        synced = True
                        print("[SYNCED] Tracking y Sync enganchados!")
                        next_sample_time = current_time + (SAMPLE_INTERVAL_MS / 1000.0)
                        
                elif synced and current_time >= next_sample_time:
                    sym_tl = COLOR_TO_SYMBOL.get(c_tl, 0)
                    sym_tr = COLOR_TO_SYMBOL.get(c_tr, 0)
                    sym_bl = COLOR_TO_SYMBOL.get(c_bl, 0)
                    sym_br = COLOR_TO_SYMBOL.get(c_br, 0)
                    
                    byte_val = (sym_tl << 6) | (sym_tr << 4) | (sym_bl << 2) | sym_br
                    received_payload.append(byte_val)
                    
                    print(f"Byte {len(received_payload)}/{EXPECTED_PAYLOAD_BYTES}: '{chr(byte_val) if 32<=byte_val<127 else hex(byte_val)}'")
                    
                    if len(received_payload) >= EXPECTED_PAYLOAD_BYTES: break
                    next_sample_time += (SAMPLE_INTERVAL_MS / 1000.0)

            else:
                cv2.putText(display, "Buscando Pantalla...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
            cv2.imshow("Receptor Dinamico", display)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')): break

    finally:
        cap.release(); cv2.destroyAllWindows()
        if received_payload:
            msg = received_payload[1:].decode('utf-8', errors='replace')
            print(f"\n=== MENSAJE ===\n{msg}\n==============")
            OUTPUT_PATH.write_bytes(received_payload[1:])

if __name__ == "__main__":
    main()