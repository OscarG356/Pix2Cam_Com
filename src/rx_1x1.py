#!/usr/bin/env python3
"""
Receptor minimalista 1x1 con sincronización por transiciones.

Usa video en vivo y detecta transiciones (cambios entre verde/negro).
Luego decodifica correctamente con timing automático.

Uso:
    python3 src/rx_1x1.py

Teclas:
    'q' = salir
"""

import cv2
import numpy as np
import time
from pathlib import Path
from typing import Optional


CAMERA_INDEX = 0
OUTPUT_PATH = Path("mensaje_recibido.txt")
SYNC_CYCLES = 1
# Flag de inicio: un frame verde y luego uno negro
SAMPLE_INTERVAL_MS = 500  # Cambia este valor para modificar el periodo de muestreo
EXPECTED_PAYLOAD_BYTES = 24



def detect_bit(frame_bgr: np.ndarray) -> Optional[int]:
    """
    Detecta si el frame es principalmente NEGRO (0) o BLANCO (1).
    
    Retorna 0 para negro, 1 para blanco, o None si no puede decidir.
    """
    # Convertir a escala de grises
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # Media de intensidad
    mean_intensity = gray.mean()

    # Threshold en 127 (mitad del rango 0-255)
    bit = 1 if mean_intensity > 127 else 0

    return bit


def detect_green(frame_bgr: np.ndarray) -> bool:
    """Detecta si el frame es predominantemente VERDE (sync).

    Usa espacio HSV y busca tonos de hue alrededor de 60 grados.
    """
    # Método más tolerante: comparar medias por canal B,G,R
    b, g, r = cv2.split(frame_bgr)
    b_mean = float(b.mean())
    g_mean = float(g.mean())
    r_mean = float(r.mean())

    # Condición: canal G significativamente mayor que R y B, y brillo suficiente
    if g_mean > 90 and (g_mean - r_mean) > 25 and (g_mean - b_mean) > 25:
        return True
    # Fallback: si la media de brillo es alta y el HSV cae en rango verde
    try:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        mask = (h >= 40) & (h <= 80) & (s > 70) & (v > 70)
        if mask.mean() > 0.4:
            return True
    except Exception:
        pass

    return False


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
    
    print("Receptor 1x1 con sincronización verde/negro")
    print("Esperando flag de inicio verde -> negro... (q=salir)")
    
    # Estado de sincronización
    sync_state = 0  # 0 = esperando verde, 1 = esperando negro, 2 = sincronizado
    last_bit: Optional[int] = None
    synced = False
    # Estado para detección de protocolo: tras sincronización empezar a leer N bytes
    expected_payload_len: Optional[int] = None
    received_payload = bytearray()
    
    # Estado de decodificación
    collected_bits = []
    collected_bytes = bytearray()
    next_sample_time = 0.0
    sample_count = 0
    start_time = time.monotonic()
    # Tiempos para detectar sync sostenido
    green_since: Optional[float] = None
    black_since: Optional[float] = None
    
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Error leyendo frame")
                continue
            
            current_time = time.monotonic()
            detected_bit = detect_bit(frame)
            is_green = detect_green(frame)
            
            if detected_bit is not None:
                # ===== FASE 1: SINCRONIZACIÓN POR FLAG VERDE -> NEGRO =====
                if not synced:
                    sample_interval_s = SAMPLE_INTERVAL_MS / 1000.0
                    # 1) detectar verde sostenido
                    if is_green:
                        if green_since is None:
                            green_since = current_time
                        # si aún estábamos contando negro, reiniciarlo
                        black_since = None
                    else:
                        # si no está en verde y no alcanzamos la duración, cancelar
                        if green_since is None or (current_time - green_since) < sample_interval_s:
                            green_since = None
                        else:
                            # verde sostenido ya detectado: ahora buscamos negro sostenido
                            if detected_bit == 0:
                                if black_since is None:
                                    black_since = current_time
                                elif (current_time - black_since) >= sample_interval_s:
                                    # Sync completo: verde 500ms seguido de negro 500ms
                                    synced = True
                                    expected_payload_len = EXPECTED_PAYLOAD_BYTES
                                    print(f"[SYNCED] Verde seguido de negro detectado en t={current_time - start_time:.3f}s")
                                    print(f"[SYNCED] Esperando {expected_payload_len} bytes de payload")
                                    print(f"[SYNCED] Muestreo cada {SAMPLE_INTERVAL_MS}ms")
                                    # Esperar 200ms tras la secuencia y luego comenzar a muestrear
                                    # en el centro de la primera celda (half-bit) para mayor robustez.
                                    sample_interval_s = SAMPLE_INTERVAL_MS / 1000.0
                                    post_sync_wait = 0.2
                                    next_sample_time = current_time + post_sync_wait + (sample_interval_s / 2)
                                    print(f"[SYNCED] Esperando {post_sync_wait*1000:.0f}ms antes de comenzar a muestrear")
                                    sample_count = 0
                                    # limpiar contadores
                                    green_since = None
                                    black_since = None
                            else:
                                # todavía no hay negro sostenido
                                black_since = None
                
                # ===== FASE 2: DECODIFICACIÓN CON TIMING CALIBRADO =====
                elif synced:
                    # Muestrear cada intervalo fijo configurable
                    sample_interval_s = SAMPLE_INTERVAL_MS / 1000.0

                    if current_time >= next_sample_time:
                        time_since_sample = current_time - (next_sample_time - sample_interval_s)
                        collected_bits.append(detected_bit)
                        sample_count += 1
                        
                        print(f"Bit {len(collected_bits)}: {detected_bit} (t={time_since_sample*1000:.1f}ms)")
                        
                        # Si completamos un byte
                        if len(collected_bits) == 8:
                            byte_val = bits_to_byte(collected_bits)
                            # Tras sincronización inmediata, coleccionamos bytes del mensaje
                            received_payload.append(byte_val)
                            printable = chr(byte_val) if 32 <= byte_val < 127 else f"0x{byte_val:02X}"
                            print(f"  → Byte {len(received_payload)}: {printable}")
                            # Si recibimos la cantidad solicitada, terminar
                            if expected_payload_len is not None and len(received_payload) >= expected_payload_len:
                                print("[PROTO] Conteo objetivo alcanzado. Finalizando recepción.")
                                collected_bytes = received_payload
                                break
                            collected_bits = []
                        
                        periods_behind = int((current_time - next_sample_time) // sample_interval_s)
                        next_sample_time += (periods_behind + 1) * sample_interval_s
                    # Registrar última muestra leída (útil para debug y para posibles detectores de borde)
                    last_bit = detected_bit
            
            # UI - mostrar múltiples líneas con estado para depuración
            display = frame.copy()
            h, w = display.shape[:2]

            if not synced:
                status_lines = [f"Sincronizando flag... estado {sync_state}/2",
                                f"last_bit={last_bit}"]
            else:
                bits_str = ''.join(str(b) for b in collected_bits) if collected_bits else ''
                status_lines = [f"SYNCED (muestreo {SAMPLE_INTERVAL_MS}ms)",
                                f"Bits: {len(collected_bits)}/8 ({bits_str})",
                                f"Bytes recibidos: {len(received_payload)}",
                                f"Samples: {sample_count}"]

            for i, line in enumerate(status_lines):
                cv2.putText(display, line, (10, 30 + i*28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Receptor 1x1", display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        if collected_bytes:
            msg = collected_bytes.decode('utf-8', errors='replace')
            print(f"\n=== MENSAJE RECIBIDO ===")
            print(msg)
            print(f"=== ({len(collected_bytes)} bytes) ===\n")
            
            OUTPUT_PATH.write_bytes(collected_bytes)
            print(f"Guardado en: {OUTPUT_PATH}")
        else:
            print("No se recibió nada.")


if __name__ == "__main__":
    main()

