#!/usr/bin/env python3
"""
Emisor 8x4 (Blanco y Negro) - 16 bits paralelos (2 Bytes por Frame)
Marco de seguimiento Verde Neón.
"""

import cv2
import numpy as np
import time
from pathlib import Path

TX_FRAME_MS = 150

class Transmitter4x4BW:
    def __init__(self, payload: bytes, frame_ms: int = TX_FRAME_MS):
        # Ahora rellenamos para que el payload sea múltiplo de 4
        faltan = len(payload) % 4
        if faltan != 0:
            payload += b'\x00' * (4 - faltan)
        self.payload = payload
        self.frame_ms = frame_ms
        self.transmitting = False

    def _get_canvas(self):
        """Lienzo Panorámico 8x4 con marco ROJO y foso negro."""
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        
        # 1. Marco Rojo exterior ancho (X: 40 a 1880, Y: 20 a 1060) -> 1840x1040
        cv2.rectangle(img, (40, 20), (1880, 1060), (0, 0, 255), -1)
        
        # 2. Foso Negro de 80px (X: 80 a 1840, Y: 60 a 1020)
        cv2.rectangle(img, (80, 60), (1840, 1020), (0, 0, 0), -1)
        
        # Movemos los textos para que se lean sobre el foso negro
        cv2.putText(img, f"Matriz 8x4: {len(self.payload)} bytes", (100, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(img, "ENTER para iniciar", (100, 1000), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        return img

    def run(self):
        # Le cambiamos el nombre a la ventana para hacerle justicia
        cv2.namedWindow('Emisor 8x4', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.setWindowProperty('Emisor 8x4', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        img = self._get_canvas()
        # ELIMINAMOS los putText viejos que estaban aquí
        cv2.imshow('Emisor 8x4', img)
        
        while True:
            key = cv2.waitKey(100) & 0xFF
            if key in (13, ord('\r')):
                if not self.transmitting:
                    self.transmitting = True
                    self._transmit()
            if key in (27, ord('q')): break
        cv2.destroyAllWindows()

    def _draw_matrix(self, bit_array: list[int]):
        img = self._get_canvas()
        # Matriz interna 8x4 (8 columnas x 4 filas) = 32 bits
        for i, bit in enumerate(bit_array):
            row = i // 8   # Dividir entre 8 columnas
            col = i % 8
            
            # El origen de la matriz ahora empieza en X=160, Y=140
            x_start = 160 + (col * 200) + 20
            y_start = 140 + (row * 200) + 20
            x_end = x_start + 160
            y_end = y_start + 160
            
            color = (255, 255, 255) if bit == 1 else (0, 0, 0)
            cv2.rectangle(img, (x_start, y_start), (x_end, y_end), color, -1)
        return img

    def _transmit(self):
        self._send_sync_sequence()
        start_time = time.monotonic()
        frames_sent = 0

        # Procesamos de a 4 bytes (32 bits)
        for i in range(0, len(self.payload), 4):
            if not self.transmitting: break
            
            byte1 = self.payload[i]
            byte2 = self.payload[i+1]
            byte3 = self.payload[i+2]
            byte4 = self.payload[i+3]
            
            # Unir en 32 bits
            word32 = (byte1 << 24) | (byte2 << 16) | (byte3 << 8) | byte4
            bits = [(word32 >> (31 - j)) & 1 for j in range(32)]

            img = self._draw_matrix(bits)
            cv2.imshow('Emisor 8x4', img) # (Mantén el nombre de ventana que tenías)
            frames_sent += 1

            target_time = start_time + (frames_sent * (self.frame_ms / 1000.0))
            if self._wait_until(target_time) in (27, ord('q')):
                self.transmitting = False
                break

        img = self._get_canvas()
        cv2.putText(img, "FIN 8x4", (750, 540), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 5)
        cv2.imshow('Emisor 8x4', img)
        self.transmitting = False

    def _send_sync_sequence(self):
        # SYNC ahora son 32 bits
        img = self._draw_matrix([1] * 32)
        cv2.imshow('Emisor 8x4', img)
        if self._wait_until(time.monotonic() + self.frame_ms/1000.0) != 255: return
        
        img = self._draw_matrix([0] * 32)
        cv2.imshow('Emisor 8x4', img)
        if self._wait_until(time.monotonic() + self.frame_ms/1000.0) != 255: return

    def _wait_until(self, target_time: float) -> int:
        while time.monotonic() < target_time:
            key = cv2.waitKey(1) & 0xFF
            if key != 255: return key
        return 255

def main():
    mensaje_path = Path("mensaje.txt")
    if not mensaje_path.is_file(): mensaje_path.write_text("¡Hola mundo en 8x4!")
    
    # Preámbulo de 4 bytes (0x55, 0x55, 0x55, 0x55)
    payload = bytes([0x55, 0x55, 0x55, 0x55]) + mensaje_path.read_bytes()
    Transmitter4x4BW(payload).run()

if __name__ == "__main__":
    main()