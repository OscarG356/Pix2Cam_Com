#!/usr/bin/env python3
"""
Receptor minimal 1x1 para Pix2Cam_Com.

Lee video en vivo, detecta si el cuadrado es NEGRO (0) o BLANCO (1).
Reconstruye los bits en bytes.

Uso:
    python3 src/rx_1x1.py [--camera 0] [--frame-ms 1000] [--debug]

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
    parser = argparse.ArgumentParser(description="Receptor 1x1 minimalista")
    parser.add_argument("--camera", type=int, default=0, help="Índice de cámara")
    parser.add_argument("--frame-ms", type=int, default=1000, help="Duración esperada del frame")
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
    
    print("Receptor 1x1")
    print("Esperando bits... (d=debug, q=salir)")
    
    debug_mode = args.debug
    collected_bits = []
    collected_bytes = bytearray()
    last_bit: Optional[int] = None
    stable_frames = 0
    last_update_time = 0.0
    sample_period = args.frame_ms / 1000.0
    
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Error leyendo frame")
                continue
            
            # Detectar bit
            detected_bit = detect_bit(frame, debug=debug_mode)
            
            if detected_bit is not None:
                now = time.monotonic()
                
                # Estabilidad: esperar a que se repita el bit
                if detected_bit == last_bit:
                    stable_frames += 1
                    
                    # Emitir si es estable y pasó suficiente tiempo
                    if stable_frames >= 2 and now >= last_update_time:
                        collected_bits.append(detected_bit)
                        
                        # Mostrar progreso
                        if len(collected_bits) <= 8:
                            print(f"Bit {len(collected_bits)}: {detected_bit}")
                        
                        # Si completamos un byte
                        if len(collected_bits) == 8:
                            byte_val = bits_to_byte(collected_bits)
                            collected_bytes.append(byte_val)
                            print(f"  → Byte {len(collected_bytes)}: 0x{byte_val:02X} ({chr(byte_val) if 32 <= byte_val < 127 else '?'})")
                            collected_bits = []
                        
                        last_update_time = now + sample_period * 0.8
                else:
                    last_bit = detected_bit
                    stable_frames = 0
            
            # UI
            display = frame.copy()
            h, w = display.shape[:2]
            
            # Mostrar estado en la esquina
            status_text = f"Bits: {len(collected_bits)}/8 | Bytes: {len(collected_bytes)}"
            cv2.putText(display, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
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
