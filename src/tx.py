#!/usr/bin/env python3
"""
Emisor 2x2 usando OpenCV - Matriz de Color Paralela
Capacidad máxima: 1 Frame = 1 Byte completo (8 bits repartidos en 4 cuadrantes).
"""

import cv2
import numpy as np
import time
from pathlib import Path

TX_FRAME_MS = 150

COLOR_MAP_4 = {
    0: (0, 0, 0),       # 00: Negro
    1: (255, 0, 0),     # 01: Azul
    2: (0, 255, 0),     # 10: Verde
    3: (0, 255, 255)    # 11: Amarillo
}

class Transmitter2x2Color:
    def __init__(self, payload: bytes, frame_ms: int = TX_FRAME_MS):
        self.payload = payload
        self.frame_ms = frame_ms
        self.transmitting = False
        self.color_map = COLOR_MAP_4
    
    def run(self):
        cv2.namedWindow('Emisor 2x2', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.setWindowProperty('Emisor 2x2', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        self._show_ready()
        
        while True:
            key = cv2.waitKey(100) & 0xFF
            if key == ord('\r') or key == 13:
                if not self.transmitting:
                    self.transmitting = True
                    self._transmit()
            if key in (27, ord('q')):
                break
        cv2.destroyAllWindows()
    
    def _show_ready(self):
        img = np.full((1080, 1920, 3), 128, dtype=np.uint8)
        cv2.putText(img, f"Modo Matriz 2x2: {len(self.payload)} bytes", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 3)
        cv2.putText(img, "Velocidad: 1 Byte por Frame (8 bits paralelos)", (100, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 255), 3)
        cv2.putText(img, "Presiona ENTER para iniciar", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 3)
        cv2.imshow('Emisor 2x2', img)
    
    def _transmit(self):
        self._send_sync_sequence(repeats=1)

        start_time = time.monotonic()
        frames_sent = 0
        total_bytes = len(self.payload)

        for byte_idx, b in enumerate(self.payload):
            if not self.transmitting:
                break

            # Extraer 2 bits para cada cuadrante mediante desplazamientos de bits
            symbol_tl = (b >> 6) & 3   # Bits 7-6
            symbol_tr = (b >> 4) & 3   # Bits 5-4
            symbol_bl = (b >> 2) & 3   # Bits 3-2
            symbol_br = b & 3          # Bits 1-0

            # Construir el lienzo de la matriz
            img = np.zeros((1080, 1920, 3), dtype=np.uint8)
            img[0:540, 0:960] = self.color_map[symbol_tl]     # Top-Left
            img[0:540, 960:1920] = self.color_map[symbol_tr]   # Top-Right
            img[540:1080, 0:960] = self.color_map[symbol_bl]   # Bottom-Left
            img[540:1080, 960:1920] = self.color_map[symbol_br] # Bottom-Right

            # Texto centrado de progreso
            progress = f"Byte {byte_idx+1}/{total_bytes} | Enviando: '{chr(b) if 32<=b<127 else hex(b)}'"
            cv2.putText(img, progress, (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

            cv2.imshow('Emisor 2x2', img)
            frames_sent += 1

            # Sincronización estricta de tiempo
            target_time = start_time + (frames_sent * (self.frame_ms / 1000.0))
            if self._wait_until(target_time) in (27, ord('q')):
                self.transmitting = False
                break

        # Marcar como completado y mostrar mensaje final
        self.bit_index_done = True
        img = np.full((1080, 1920, 3), 128, dtype=np.uint8)
        cv2.putText(img, "TRANSMISION COMPLETADA 2x2", (150, 400), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 4)
        cv2.imshow('Emisor 2x2', img)

        self.transmitting = False

    def _send_sync_sequence(self, repeats: int = 1):
        """Pone toda la matriz en ROJO y luego todo en NEGRO."""
        for i in range(repeats):
            for color in [(0, 0, 255), (0, 0, 0)]: # Rojo, luego Negro
                img = np.zeros((1080, 1920, 3), dtype=np.uint8)
                img[0:540, 0:960] = color
                img[0:540, 960:1920] = color
                img[540:1080, 0:960] = color
                img[540:1080, 960:1920] = color
                cv2.imshow('Emisor 2x2', img)
                if self._wait_fixed_ms(self.frame_ms) in (27, ord('q')):
                    self.transmitting = False
                    return

    def _wait_fixed_ms(self, ms: int) -> int:
        start = time.monotonic()
        while (time.monotonic() - start) < (ms / 1000.0):
            key = cv2.waitKey(1) & 0xFF
            if key != 255: return key
        return 255

    def _wait_until(self, target_time: float) -> int:
        while time.monotonic() < target_time:
            key = cv2.waitKey(1) & 0xFF
            if key != 255: return key
        return 255

def main():
    mensaje_path = Path("mensaje.txt")
    if not mensaje_path.is_file():
        mensaje_path.write_text("hola amigos de youtube")
    
    payload = mensaje_path.read_bytes()
    sync_preamble = bytes([0x55]) # Nuestra fiel letra 'U'
    full_payload = sync_preamble + payload
    
    print(f"Iniciando Transmisor Matriz 2x2 ({len(full_payload)} Bytes)")
    tx = Transmitter2x2Color(full_payload)
    tx.run()

if __name__ == "__main__":
    main()