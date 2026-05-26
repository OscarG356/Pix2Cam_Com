#!/usr/bin/env python3
"""
Receptor minimalista 1x1 con sincronización por transiciones.

Lee video en vivo y detecta transiciones (cambios entre negro/blanco).
Las primeras 5 transiciones permiten calibrar la duración del bit.
Luego decodifica correctamente con timing automático.

Uso:
    python3 src/rx_1x1.py [--camera 0] [--debug]

Teclas:
    'd' = toggle debug
    'q' = salir
"""

import argparse
import cv2
import numpy as np
import time
from collections import deque
from pathlib import Path
from typing import Optional


def detect_bit(frame_bgr: np.ndarray, debug: bool = False) -> Optional[int]:
    """
    Detecta si el frame es principalmente NEGRO (0) o BLANCO (1).
    
    Retorna 0 para negro, 1 para blanco, o None si no puede decidir.
    """
    # Convertir a escala de grises
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    
    # Media de intensidad
    mean_intensity = gray.mean()
    
    if debug:
        print(f"[DEBUG] Mean intensity: {mean_intensity:.1f}")
    
    # Threshold en 127 (mitad del rango 0-255)
    bit = 1 if mean_intensity > 127 else 0
    
    if debug:
        print(f"[DEBUG] Bit: {bit} ({'WHITE' if bit else 'BLACK'})")
    
    return bit


def bits_to_byte(bits: list[int]) -> int:
    """Convierte 8 bits a un byte (MSB primero)."""
    byte_val = 0
    for i, bit in enumerate(bits):
        byte_val |= (bit << (7 - i))
    return byte_val


def main():
    parser = argparse.ArgumentParser(description="Receptor 1x1 con sincronización por transiciones")
    parser.add_argument("--camera", type=int, default=0, help="Índice de cámara")
    parser.add_argument("--debug", action="store_true", help="Modo debug")
    parser.add_argument("--output", type=Path, help="Archivo de salida")
    
    args = parser.parse_args()
    
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: no se pudo abrir cámara {args.camera}")
        return
    
    # Configuración de cámara
    cap.set(cv2.CAP_PROP_EXPOSURE, -15)  # Reducir exposición
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Mantener solo el frame más reciente
    
    print("Receptor 1x1 con sincronización por transiciones")
    print("Esperando transiciones para calibrar... (d=debug, q=salir)")
    
    debug_mode = args.debug
    
    # Estado de sincronización
    transition_times = []  # Tiempos de transiciones
    bit_duration = None  # Duración calculada del bit
    last_bit: Optional[int] = None
    synced = False
    
    # Estado de decodificación
    collected_bits = []
    collected_bytes = bytearray()
    last_sample_time = 0.0
    sample_count = 0
    start_time = time.monotonic()
    
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Error leyendo frame")
                continue
            
            current_time = time.monotonic()
            detected_bit = detect_bit(frame, debug=debug_mode)
            
            if detected_bit is not None:
                # ===== FASE 1: SINCRONIZACIÓN POR TRANSICIONES =====
                if not synced and len(transition_times) < 5:
                    # Detectar transición (cambio de bit)
                    if last_bit is not None and detected_bit != last_bit:
                        transition_times.append(current_time)
                        print(f"[Transición {len(transition_times)}] en t={current_time - start_time:.3f}s")
                        
                        # Si tenemos 5 transiciones, calcular duración del bit
                        if len(transition_times) == 5:
                            # Calcular intervalos entre transiciones
                            intervals = [transition_times[i+1] - transition_times[i] 
                                       for i in range(len(transition_times)-1)]
                            bit_duration = np.mean(intervals)
                            print(f"\n[SYNCED] Duración calibrada del bit: {bit_duration*1000:.1f}ms")
                            print(f"Intervalos: {[f'{x*1000:.1f}ms' for x in intervals]}")
                            synced = True
                            last_sample_time = transition_times[-1]
                            sample_count = 0
                    
                    last_bit = detected_bit
                
                # ===== FASE 2: DECODIFICACIÓN CON TIMING CALIBRADO =====
                elif synced and bit_duration is not None:
                    # Muestrear cada bit_duration
                    time_since_sync = current_time - last_sample_time
                    
                    if time_since_sync >= bit_duration * 0.9:  # 90% del período
                        collected_bits.append(detected_bit)
                        sample_count += 1
                        
                        print(f"Bit {len(collected_bits)}: {detected_bit} (t={time_since_sync*1000:.1f}ms)")
                        
                        # Si completamos un byte
                        if len(collected_bits) == 8:
                            byte_val = bits_to_byte(collected_bits)
                            collected_bytes.append(byte_val)
                            printable = chr(byte_val) if 32 <= byte_val < 127 else f"0x{byte_val:02X}"
                            print(f"  → Byte {len(collected_bytes)}: {printable}\n")
                            collected_bits = []
                        
                        last_sample_time = current_time
            
            # UI
            display = frame.copy()
            h, w = display.shape[:2]
            
            if not synced:
                status = f"Sincronizando... {len(transition_times)}/5 transiciones"
            else:
                status = f"Bits: {len(collected_bits)}/8 | Bytes: {len(collected_bytes)}"
            
            cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Receptor 1x1", display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                debug_mode = not debug_mode
                print(f"[DEBUG] Mode: {'ON' if debug_mode else 'OFF'}")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        if collected_bytes:
            msg = collected_bytes.decode('utf-8', errors='replace')
            print(f"\n=== MENSAJE RECIBIDO ===")
            print(msg)
            print(f"=== ({len(collected_bytes)} bytes) ===\n")
            
            if args.output:
                args.output.write_bytes(collected_bytes)
                print(f"Guardado en: {args.output}")
        else:
            print("No se recibió nada.")


if __name__ == "__main__":
    main()

