#!/usr/bin/env python3
"""
Emisor 1x2 usando OpenCV - Multiplexación Espacial (Pantalla Dividida)
Capacidad duplicada: 4 bits por frame (2 bits lado izquierdo, 2 bits lado derecho).
"""

import cv2
import numpy as np
import time
from pathlib import Path

# Velocidad de transmisión en milisegundos
TX_FRAME_MS = 150

# Mapa de colores BGR de OpenCV (4 colores = 2 bits)
COLOR_MAP_4 = {
    0: (0, 0, 0),       # 00: Negro
    1: (255, 0, 0),     # 01: Azul
    2: (0, 255, 0),     # 10: Verde
    3: (0, 255, 255)    # 11: Amarillo
}

class Transmitter1x2Color:
    def __init__(self, payload: bytes, frame_ms: int = TX_FRAME_MS):
        self.payload = payload
        self.frame_ms = frame_ms
        self.transmitting = False
        self.bit_index = 0
        self.bits_per_frame = 4  # 2 bits izquierda + 2 bits derecha
        self.color_map = COLOR_MAP_4
    
    def run(self):
        cv2.namedWindow('Emisor 1x2', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.setWindowProperty('Emisor 1x2', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        self._show_ready()
        
        while True:
            key = cv2.waitKey(100) & 0xFF
            if key == ord('\r') or key == 13:
                if not self.transmitting:
                    self.transmitting = True
                    self.bit_index = 0
                    self._transmit()
            
            if key in (27, ord('q')):
                break
        
        cv2.destroyAllWindows()
    
    def _show_ready(self):
        img = np.full((1080, 1920, 3), 128, dtype=np.uint8)
        total_bits = len(self.payload) * 8
        cv2.putText(img, f"Modo 1x2: {len(self.payload)} bytes = {total_bits} bits", 
                   (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 3)
        cv2.putText(img, f"Capacidad: {self.bits_per_frame} bits/frame (Pantalla Dividida)", 
                   (100, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 255), 3)
        cv2.putText(img, "Presiona ENTER para iniciar", 
                   (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 3)
        cv2.imshow('Emisor 1x2', img)
    
    def _transmit(self):
        # Sincronizamos ambos lados en Rojo -> Negro simultáneamente
        self._send_sync_sequence(repeats=1)

        total_bits = len(self.payload) * 8
        start_time = time.monotonic()
        frames_sent = 0

        while self.transmitting and self.bit_index < total_bits:
            # Leer hasta 4 bits del flujo de datos
            bits_to_read = min(self.bits_per_frame, total_bits - self.bit_index)
            chunk = 0
            
            for i in range(bits_to_read):
                current_bit_pos = self.bit_index + i
                byte_idx = current_bit_pos // 8
                bit_in_byte = 7 - (current_bit_pos % 8)
                bit = (self.payload[byte_idx] >> bit_in_byte) & 1
                chunk = (chunk << 1) | bit
            
            # Rellenar con ceros a la derecha si es el último frame y faltan bits
            if bits_to_read < self.bits_per_frame:
                chunk = chunk << (self.bits_per_frame - bits_to_read)

            # Separar el chunk de 4 bits en dos símbolos de 2 bits cada uno
            symbol_left = (chunk >> 2) & 3   # Bits posiciones altas (0 y 1)
            symbol_right = chunk & 3         # Bits posiciones bajas (2 y 3)

            # Construir el frame dividido en dos mitades (960 píxeles de ancho cada una)
            img = np.zeros((1080, 1920, 3), dtype=np.uint8)
            img[:, :960] = self.color_map[symbol_left]   # Mitad Izquierda
            img[:, 960:] = self.color_map[symbol_right]  # Mitad Derecha

            # Texto informativo de depuración superpuesto en el centro
            progress = f"Bit: {self.bit_index + bits_to_read}/{total_bits} | L:[{bin(symbol_left)[2:].zfill(2)}] R:[{bin(symbol_right)[2:].zfill(2)}]"
            cv2.putText(img, progress, (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

            cv2.imshow('Emisor 1x2', img)

            self.bit_index += bits_to_read
            frames_sent += 1

            # Control adaptativo estricto del tiempo
            target_time = start_time + (frames_sent * (self.frame_ms / 1000.0))
            key = self._wait_until(target_time)
            if key in (27, ord('q')):
                self.transmitting = False
                break

        if self.bit_index >= total_bits:
            img = np.full((1080, 1920, 3), 128, dtype=np.uint8)
            cv2.putText(img, "TRANSMISION COMPLETADA", (200, 400),
                       cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 4)
            cv2.imshow('Emisor 1x2', img)

        self.transmitting = False

    def _send_sync_sequence(self, repeats: int = 1):
        """Enciende y apaga ambas mitades coordinadas para que el receptor enganche el inicio."""
        width_half = 960
        for i in range(repeats):
            # Estado 1: Todo ROJO
            img = np.zeros((1080, 1920, 3), dtype=np.uint8)
            img[:, :width_half] = (0, 0, 255)
            img[:, width_half:] = (0, 0, 255)
            cv2.putText(img, f"SYNC ROJO ({i+1})", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            cv2.imshow('Emisor 1x2', img)
            if (self._wait_fixed_ms(self.frame_ms) in (27, ord('q'))): return

            # Estado 2: Todo NEGRO
            img = np.zeros((1080, 1920, 3), dtype=np.uint8)
            cv2.putText(img, f"SYNC NEGRO ({i+1})", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            cv2.imshow('Emisor 1x2', img)
            if (self._wait_fixed_ms(self.frame_ms) in (27, ord('q'))): return

    def _wait_fixed_ms(self, ms: int) -> int:
        start = time.monotonic()
        while (time.monotonic() - start) < (ms / 1000.0):
            key = cv2.waitKey(1) & 0xFF
            if key != 255: return key
        return 255

    def _wait_until(self, target_time: float) -> int:
        while time.monotonic() < target_time:
            key = cv2.waitKey(1) & 0xFF
            if key != 255:
                return key
        return 255

def main():
    mensaje_path = Path("mensaje.txt")
    if not mensaje_path.is_file():
        mensaje_path.write_text("hola amigos de youtube")
    
    payload = mensaje_path.read_bytes()
    if not payload:
        print("Error: archivo vacío")
        return
    
    sync_preamble = bytes([0x55]) # El byte 'U' para mantener compatibilidad
    full_payload = sync_preamble + payload
    
    print(f"Transmitiendo en modo 1x2... Tamaño: {len(full_payload)} bytes")
    tx = Transmitter1x2Color(full_payload, frame_ms=TX_FRAME_MS)
    tx.run()

if __name__ == "__main__":
    main()