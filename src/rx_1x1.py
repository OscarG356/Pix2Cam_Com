#!/usr/bin/env python3
"""
Receptor 1x1 con modulación por 4 colores (CSK - Color Shift Keying).

Decodifica 2 bits por frame basándose en el color detectado:
- NEGRO    (00) -> 0
- AZUL     (01) -> 1
- VERDE    (10) -> 2
- AMARILLO (11) -> 3

Sincronización: Busca ROJO continuo seguido de NEGRO continuo.
"""

import cv2
import numpy as np
import time
from pathlib import Path
from typing import Optional

CAMERA_INDEX = 0
OUTPUT_PATH = Path("mensaje_recibido.txt")

# Debe coincidir con el TX_FRAME_MS del emisor
SAMPLE_INTERVAL_MS = 500
EXPECTED_PAYLOAD_BYTES = 432

# Mapeo de colores a símbolos (valores numéricos de 2 bits)
COLOR_TO_SYMBOL = {
    "BLACK": 0,  # 00
    "BLUE": 1,   # 01
    "GREEN": 2,  # 10
    "YELLOW": 3  # 11
}

def detect_dominant_color(frame_bgr: np.ndarray) -> tuple[str, float]:
    """
    Analiza el centro del frame en espacio HSV para determinar el color dominante.
    Retorna "BLACK", "RED", "BLUE", "GREEN", "YELLOW" o "UNKNOWN".
    """
    # Recortar el 50% central para evitar capturar bordes de la pantalla o la habitación
    h, w = frame_bgr.shape[:2]
    roi = frame_bgr[h//4 : 3*h//4, w//4 : 3*w//4]
    
    # Convertir a HSV para ser más resilientes a cambios de iluminación
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Definir rangos HSV (Hue: 0-179, Saturation: 0-255, Value: 0-255)
    # Negro: Valor (brillo) bajo
    mask_black = cv2.inRange(hsv, (0, 0, 0), (179, 255, 80))
    
    # Rojo: El Hue en rojo envuelve los extremos (0-10 y 160-179)
    mask_red1 = cv2.inRange(hsv, (0, 70, 80), (10, 255, 255))
    mask_red2 = cv2.inRange(hsv, (160, 70, 80), (179, 255, 255))
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    
    # Azul
    mask_blue = cv2.inRange(hsv, (90, 70, 80), (135, 255, 255))
    
    # Verde
    mask_green = cv2.inRange(hsv, (40, 70, 80), (85, 255, 255))
    
    # Amarillo
    mask_yellow = cv2.inRange(hsv, (15, 70, 80), (35, 255, 255))
    
    # Calcular el porcentaje de píxeles que cumplen cada máscara
    area = roi.shape[0] * roi.shape[1]
    
    scores = {
        "BLACK": cv2.countNonZero(mask_black) / area,
        "RED": cv2.countNonZero(mask_red) / area,
        "BLUE": cv2.countNonZero(mask_blue) / area,
        "GREEN": cv2.countNonZero(mask_green) / area,
        "YELLOW": cv2.countNonZero(mask_yellow) / area,
    }
    
    # Encontrar el color con mayor porcentaje de coincidencia
    best_color = max(scores, key=scores.get)
    
    # Devolver el mejor color y su fracción (permitimos decidir fuera de la función)
    best_score = scores[best_color]
    if best_score > 0.0:
        return best_color, best_score

    return "UNKNOWN", 0.0

def bits_to_byte(bits: list[int]) -> int:
    """Convierte 8 bits a un byte (MSB primero)."""
    byte_val = 0
    for i, bit in enumerate(bits):
        byte_val |= (bit << (7 - i))
    return byte_val

def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: no se pudo abrir cámara {CAMERA_INDEX}")
        return
    
    print("Receptor 1x1 Color - 2 bits/simbolo")
    print("Esperando flag de sincronización: ROJO -> NEGRO... (q=salir)")
    
    synced = False
    received_payload = bytearray()
    collected_bits = []
    
    next_sample_time = 0.0
    sample_count = 0
    start_time = time.monotonic()
    
    # Contadores para detectar sync por frames consecutivos
    SYNC_RED_FRAMES = 2
    SYNC_BLACK_FRAMES = 2
    COLOR_ACCEPT_THRESHOLD = 0.15

    red_count = 0
    black_count = 0
    red_locked = False
    
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            
            current_time = time.monotonic()
            detected_color, detected_score = detect_dominant_color(frame)
            
            # ===== FASE 1: SINCRONIZACIÓN POR FLAG ROJO -> NEGRO =====
            if not synced:
                if detected_color == "RED" and detected_score >= COLOR_ACCEPT_THRESHOLD:
                    red_count += 1
                    black_count = 0
                elif detected_color == "BLACK" and detected_score >= COLOR_ACCEPT_THRESHOLD:
                    if red_count >= SYNC_RED_FRAMES:
                        black_count += 1
                    else:
                        red_count = 0
                        black_count = 0
                else:
                    if not red_locked:
                        red_count = 0
                    black_count = 0

                if red_count >= SYNC_RED_FRAMES and black_count >= SYNC_BLACK_FRAMES:
                    synced = True
                    print(f"[SYNCED] Secuencia ROJO->NEGRO detectada.")
                    
                    # --- CORRECCIÓN DE TIMING ---
                    sample_interval_s = SAMPLE_INTERVAL_MS / 1000.0
                    
                    # Estimamos cuántos ms de NEGRO ya se consumieron en la cámara
                    # (Aproximadamente: black_count * tiempo_de_un_frame_de_camara)
                    # Asumiendo una webcam estándar de 30fps (~33ms por frame):
                    tiempo_consumido_negro = black_count * 0.033
                    
                    # El tiempo que le queda al frame NEGRO actual para terminar es:
                    tiempo_restante_negro = max(0.0, sample_interval_s - tiempo_consumido_negro)
                    
                    # Queremos saltar lo que le queda al NEGRO + la MITAD del primer frame de datos
                    next_sample_time = current_time + tiempo_restante_negro + (sample_interval_s / 2)
                    
                    print(f"[TIMING] Primer muestreo programado en +{(tiempo_restante_negro + (sample_interval_s / 2))*1000:.0f}ms")
                    
                    red_count = 0
                    black_count = 0

            # ===== FASE 2: DECODIFICACIÓN DE 2 BITS POR FRAME =====
            elif synced:
                sample_interval_s = SAMPLE_INTERVAL_MS / 1000.0

                if current_time >= next_sample_time:
                    # Traducir el color leído a su valor numérico de símbolo
                    symbol = COLOR_TO_SYMBOL.get(detected_color, 0)  # Por defecto a negro si hay duda
                    
                    # Extraer los 2 bits del símbolo (MSB primero)
                    bit1 = (symbol >> 1) & 1
                    bit2 = symbol & 1
                    
                    collected_bits.extend([bit1, bit2])
                    sample_count += 1
                    
                    print(f"Frame {sample_count}: Color {detected_color} ({detected_score:.2f}) -> Simbolo {symbol} (Bits: {bit1}{bit2})")
                    
                    # Si completamos un byte (8 bits)
                    if len(collected_bits) >= 8:
                        # Extraer exactamente 8 bits (por si acaso nos pasamos, aunque aquí avanzamos de 2 en 2)
                        byte_bits = collected_bits[:8]
                        collected_bits = collected_bits[8:]
                        
                        byte_val = bits_to_byte(byte_bits)
                        received_payload.append(byte_val)
                        
                        printable = chr(byte_val) if 32 <= byte_val < 127 else f"0x{byte_val:02X}"
                        print(f"  → Byte {len(received_payload)}: {printable} ({byte_bits})")
                        
                        if len(received_payload) >= EXPECTED_PAYLOAD_BYTES:
                            print("[PROTO] Carga útil completa. Finalizando recepción.")
                            break
                    
                    periods_behind = int((current_time - next_sample_time) // sample_interval_s)
                    next_sample_time += (periods_behind + 1) * sample_interval_s
            
            # ===== INTERFAZ VISUAL PARA DEPURACIÓN =====
            display = frame.copy()
            # Dibujar un rectángulo mostrando la zona de muestreo central
            h, w = display.shape[:2]
            cv2.rectangle(display, (w//4, h//4), (3*w//4, 3*h//4), (255, 255, 255), 2)

            if not synced:
                status_lines = ["Buscando Sync (ROJO -> NEGRO)", f"Color actual: {detected_color} {f'({detected_score:.2f})' if isinstance(detected_score, float) else ''}"]
            else:
                bits_str = ''.join(str(b) for b in collected_bits)
                status_lines = [
                    f"SYNCED - Leyendo Data",
                    f"Color: {detected_color} {f'({detected_score:.2f})' if isinstance(detected_score, float) else ''}",
                    f"Bits en buffer: {bits_str}",
                    f"Bytes recibidos: {len(received_payload)}/{EXPECTED_PAYLOAD_BYTES}"
                ]

            for i, line in enumerate(status_lines):
                cv2.putText(display, line, (10, 30 + i*28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow("Receptor 1x1 Color", display)
            
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        if received_payload:
            msg = received_payload.decode('utf-8', errors='replace')
            print(f"\n=== MENSAJE RECIBIDO ({len(received_payload)} bytes) ===")
            print(msg)
            print("=====================\n")
            OUTPUT_PATH.write_bytes(received_payload)
        else:
            print("No se completó la recepción de datos.")

if __name__ == "__main__":
    main()