#!/usr/bin/env python3
"""
Receptor 1x2 usando OpenCV - Demodulación Espacial por Color
Lee dos canales simultáneos (Izquierda y Derecha) -> 4 bits por frame.
Alineado con el Emisor de alta velocidad (150ms).
"""

import cv2
import numpy as np
import time
from pathlib import Path

CAMERA_INDEX = 0
OUTPUT_PATH = Path("mensaje_recibido.txt")

# --- AJUSTE VALORES CRÍTICOS ---
SAMPLE_INTERVAL_MS = 150       # ¡Debe ser idéntico al Emisor (150ms)!
EXPECTED_PAYLOAD_BYTES = 55    # Cambiar a 510 si vas a transmitir la canción completa

COLOR_TO_SYMBOL = {
    "BLACK": 0,  # 00
    "BLUE": 1,   # 01
    "GREEN": 2,  # 10
    "YELLOW": 3  # 11
}

def detect_dominant_color(roi_frame: np.ndarray) -> tuple[str, float]:
    """Analiza una región específica y devuelve el color dominante."""
    hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
    
    mask_black = cv2.inRange(hsv, (0, 0, 0), (179, 255, 75))
    mask_red1 = cv2.inRange(hsv, (0, 70, 70), (10, 255, 255))
    mask_red2 = cv2.inRange(hsv, (160, 70, 70), (179, 255, 255))
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_blue = cv2.inRange(hsv, (90, 70, 70), (135, 255, 255))
    mask_green = cv2.inRange(hsv, (40, 70, 70), (85, 255, 255))
    mask_yellow = cv2.inRange(hsv, (15, 70, 70), (35, 255, 255))
    
    area = roi_frame.shape[0] * roi_frame.shape[1]
    scores = {
        "BLACK": cv2.countNonZero(mask_black) / area,
        "RED": cv2.countNonZero(mask_red) / area,
        "BLUE": cv2.countNonZero(mask_blue) / area,
        "GREEN": cv2.countNonZero(mask_green) / area,
        "YELLOW": cv2.countNonZero(mask_yellow) / area,
    }
    
    best_color = max(scores, key=scores.get)
    best_score = scores[best_color]
    if best_score > 0.0:
        return best_color, best_score
    return "UNKNOWN", 0.0

def bits_to_byte(bits: list[int]) -> int:
    byte_val = 0
    for i, bit in enumerate(bits):
        byte_val |= (bit << (7 - i))
    return byte_val

def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: no se pudo abrir la cámara {CAMERA_INDEX}")
        return
    
    print(f"Receptor 1x2 Activo - Esperando señal en paralelo ({SAMPLE_INTERVAL_MS}ms)...")
    
    synced = False
    received_payload = bytearray()
    collected_bits = []
    next_sample_time = 0.0
    sample_count = 0
    
    SYNC_RED_FRAMES = 2
    SYNC_BLACK_FRAMES = 2
    COLOR_ACCEPT_THRESHOLD = 0.15

    red_count = 0
    black_count = 0
    
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            
            current_time = time.monotonic()
            h, w = frame.shape[:2]
            mid_x = w // 2
            
            # --- DEFINICIÓN DE ROIS (Izquierda y Derecha) ---
            # Recortamos el centro de la mitad izquierda y el centro de la mitad derecha
            roi_left = frame[h//4 : 3*h//4, mid_x//4 : 3*mid_x//4]
            roi_right = frame[h//4 : 3*h//4, mid_x + mid_x//4 : mid_x + 3*mid_x//4]
            
            # Detectar colores en ambas zonas
            color_l, score_l = detect_dominant_color(roi_left)
            color_r, score_r = detect_dominant_color(roi_right)
            
            # ===== FASE 1: SINCRONIZACIÓN SIMULTÁNEA =====
            if not synced:
                # El emisor pone ambas pantallas en ROJO y luego ambas en NEGRO
                if color_l == "RED" and color_r == "RED" and score_l >= COLOR_ACCEPT_THRESHOLD:
                    red_count += 1
                    black_count = 0
                elif color_l == "BLACK" and color_r == "BLACK" and score_l >= COLOR_ACCEPT_THRESHOLD:
                    if red_count >= SYNC_RED_FRAMES:
                        black_count += 1
                    else:
                        red_count = 0
                        black_count = 0
                else:
                    black_count = 0

                if red_count >= SYNC_RED_FRAMES and black_count >= SYNC_BLACK_FRAMES:
                    synced = True
                    print(f"[SYNCED] ¡Sincronización espacial 1x2 exitosa!")
                    
                    sample_interval_s = SAMPLE_INTERVAL_MS / 1000.0
                    tiempo_consumido_negro = black_count * 0.033
                    tiempo_restante_negro = max(0.0, sample_interval_s - tiempo_consumido_negro)
                    
                    # Ajuste de fase para muestrear justo en la mitad del primer frame de datos reales
                    next_sample_time = current_time + tiempo_restante_negro + (sample_interval_s / 2)
                    red_count = 0
                    black_count = 0
            
            # ===== FASE 2: DECODIFICACIÓN PARALELA (4 BITS) =====
            elif synced:
                sample_interval_s = SAMPLE_INTERVAL_MS / 1000.0

                if current_time >= next_sample_time:
                    # Traducir colores a símbolos de 2 bits
                    symbol_l = COLOR_TO_SYMBOL.get(color_l, 0)
                    symbol_r = COLOR_TO_SYMBOL.get(color_r, 0)
                    
                    # Extraer los bits individuales
                    bit0 = (symbol_l >> 1) & 1
                    bit1 = symbol_l & 1
                    bit2 = (symbol_r >> 1) & 1
                    bit3 = symbol_r & 1
                    
                    # Registrar los 4 bits en orden en el flujo principal
                    collected_bits.extend([bit0, bit1, bit2, bit3])
                    sample_count += 1
                    
                    print(f"Frame {sample_count}: L:[{color_l}] R:[{color_r}] -> Bits: {bit0}{bit1}{bit2}{bit3}")
                    
                    # Procesar si acumulamos bytes completos (8 bits)
                    while len(collected_bits) >= 8:
                        byte_bits = collected_bits[:8]
                        collected_bits = collected_bits[8:]
                        
                        byte_val = bits_to_byte(byte_bits)
                        received_payload.append(byte_val)
                        
                        printable = chr(byte_val) if 32 <= byte_val < 127 else f"0x{byte_val:02X}"
                        print(f"  → Byte {len(received_payload)}/{EXPECTED_PAYLOAD_BYTES}: {printable}")
                        
                        if len(received_payload) >= EXPECTED_PAYLOAD_BYTES:
                            print("[PROTO] Carga útil completa. Finalizando recepción multiplexada.")
                            break
                    
                    if len(received_payload) >= EXPECTED_PAYLOAD_BYTES:
                        break
                    
                    # Avanzar el reloj adaptativo
                    periods_behind = int((current_time - next_sample_time) // sample_interval_s)
                    next_sample_time += (periods_behind + 1) * sample_interval_s
            
            # ===== INTERFAZ VISUAL (Guías de las dos mitades) =====
            display = frame.copy()
            
            # Dibujar rectángulos blancos sobre las zonas de lectura de la cámara
            cv2.rectangle(display, (mid_x//4, h//4), (3*mid_x//4, 3*h//4), (255, 255, 255), 2)
            cv2.rectangle(display, (mid_x + mid_x//4, h//4), (mid_x + 3*mid_x//4, 3*h//4), (255, 255, 255), 2)
            # Línea divisoria central
            cv2.line(display, (mid_x, 0), (mid_x, h), (0, 0, 255), 1)

            if not synced:
                status = "Buscando Sync 1x2 (Ambos ROJO -> Ambos NEGRO)"
                cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                cv2.putText(display, "LEYENDO DATOS (MODO 1x2)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(display, f"Bytes: {len(received_payload)}/{EXPECTED_PAYLOAD_BYTES}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("Receptor 1x2 Espacial", display)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        if received_payload:
            # Limpieza: Descartamos el primer byte (0x55 / la letra 'U') que es la sincronización
            preamble = received_payload[0]
            mensaje_puro = received_payload[1:]
            
            try:
                msg = mensaje_puro.decode('utf-8')
            except Exception:
                msg = mensaje_puro.decode('utf-8', errors='replace')
                
            print(f"\n=== MENSAJE 1x2 PROCESADO ===")
            print(f"Preámbulo descartado: 0x{preamble:02X} ('U')")
            print(f"Texto limpio obtenido: {msg}")
            print("=============================\n")
            OUTPUT_PATH.write_bytes(mensaje_puro)
        else:
            print("No se logró recolectar información.")

if __name__ == "__main__":
    main()