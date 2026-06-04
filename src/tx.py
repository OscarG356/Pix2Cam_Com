#!/usr/bin/env python3
"""
Emisor 4x4 (Blanco y Negro) - 16 bits paralelos (2 Bytes por Frame)
Marco de seguimiento Verde Neón.
"""

import cv2
import numpy as np
import time
from pathlib import Path

TX_FRAME_MS = 150

class Transmitter4x4BW:
    def __init__(self, payload: bytes, frame_ms: int = TX_FRAME_MS):
        # Asegurarnos de que el payload sea par (para mandar 2 bytes siempre)
        if len(payload) % 2 != 0:
            payload += b'\x00'
        self.payload = payload
        self.frame_ms = frame_ms
        self.transmitting = False

    def _get_canvas(self):
        """Lienzo con Marco Rojo y un 'foso' de aislamiento negro."""
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        # 1. Marco Rojo exterior (Tamaño total: 1040x1040)
        cv2.rectangle(img, (440, 20), (1480, 1060), (0, 0, 255), -1)
        # 2. "Foso" Negro de aislamiento (Borde negro de 80px de grosor)
        cv2.rectangle(img, (480, 60), (1440, 1020), (0, 0, 0), -1)
        # (La matriz de datos se dibujará automáticamente dentro, de 560 a 1360)
        return img

    def run(self):
        cv2.namedWindow('Emisor 4x4', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.setWindowProperty('Emisor 4x4', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        img = self._get_canvas()
        cv2.putText(img, f"Matriz 4x4 B/W: {len(self.payload)} bytes", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.putText(img, "ENTER para iniciar", (50, 1050), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.imshow('Emisor 4x4', img)
        
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
        # Matriz interna es de 800x800 px (X: 560-1360, Y: 140-940)
        # Cada celda es de 200x200, pero dibujamos un cuadrado interno de 160x160 para dejar padding negro
        for i, bit in enumerate(bit_array):
            row = i // 4
            col = i % 4
            
            x_start = 560 + (col * 200) + 20
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

        # Procesamos de a 2 bytes (16 bits)
        for i in range(0, len(self.payload), 2):
            if not self.transmitting: break
            
            byte1 = self.payload[i]
            byte2 = self.payload[i+1]
            
            # Unir en 16 bits y extraerlos en una lista
            word16 = (byte1 << 8) | byte2
            bits = [(word16 >> (15 - j)) & 1 for j in range(16)]

            img = self._draw_matrix(bits)
            cv2.imshow('Emisor 4x4', img)
            frames_sent += 1

            target_time = start_time + (frames_sent * (self.frame_ms / 1000.0))
            if self._wait_until(target_time) in (27, ord('q')):
                self.transmitting = False
                break

        img = self._get_canvas()
        cv2.putText(img, "FIN 4x4", (850, 540), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 5)
        cv2.imshow('Emisor 4x4', img)
        self.transmitting = False

    def _send_sync_sequence(self):
        # SYNC 1: Todo Blanco (16 bits en 1)
        img = self._draw_matrix([1] * 16)
        cv2.imshow('Emisor 4x4', img)
        if self._wait_until(time.monotonic() + self.frame_ms/1000.0) != 255: return
        
        # SYNC 2: Todo Negro (16 bits en 0)
        img = self._draw_matrix([0] * 16)
        cv2.imshow('Emisor 4x4', img)
        if self._wait_until(time.monotonic() + self.frame_ms/1000.0) != 255: return

    def _wait_until(self, target_time: float) -> int:
        while time.monotonic() < target_time:
            key = cv2.waitKey(1) & 0xFF
            if key != 255: return key
        return 255

def main():
    mensaje_path = Path("mensaje.txt")
    if not mensaje_path.is_file(): mensaje_path.write_text("¡Hola mundo en 4x4!")
    
    # Preámbulo de 2 bytes (0x55 0x55)
    payload = bytes([0x55, 0x55]) + mensaje_path.read_bytes()
    Transmitter4x4BW(payload).run()

if __name__ == "__main__":
    main()