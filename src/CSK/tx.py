#!/usr/bin/env python3
"""
Emisor 8x4 a Color - 64 bits paralelos (8 Bytes por Frame)
Colores: Negro (00), Verde (01), Rojo (10), Blanco (11)
"""

import cv2
import numpy as np
import time
from pathlib import Path

TX_FRAME_MS = 200

class Transmitter8x4Color:
    def __init__(self, payload: bytes, frame_ms: int = TX_FRAME_MS):
        # Rellenamos para que el payload sea múltiplo de 8 bytes (64 bits)
        faltan = len(payload) % 8
        if faltan != 0:
            payload += b'\x00' * (8 - faltan)
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
        
        cv2.putText(img, f"Matriz 8x4 Color: {len(self.payload)} bytes", (100, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(img, "ENTER para iniciar", (100, 1000), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        return img

    def run(self):
        cv2.namedWindow('Emisor 8x4 Color', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.setWindowProperty('Emisor 8x4 Color', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        img = self._get_canvas()
        cv2.imshow('Emisor 8x4 Color', img)
        
        while True:
            key = cv2.waitKey(100) & 0xFF
            if key in (13, ord('\r')):
                if not self.transmitting:
                    self.transmitting = True
                    self._transmit()
            if key in (27, ord('q')): break
        cv2.destroyAllWindows()

    def _draw_matrix(self, pairs_array: list[int]):
        img = self._get_canvas()
        # pairs_array tiene 32 elementos, cada uno con valores de 0 a 3 (2 bits)
        for i, val in enumerate(pairs_array):
            row = i // 8
            col = i % 8
            
            x_start = 160 + (col * 200) + 20
            y_start = 140 + (row * 200) + 20
            x_end = x_start + 160
            y_end = y_start + 160
            
            # Asignación de colores en BGR
            if val == 0:    color = (0, 0, 0)       # Negro (00)
            elif val == 1:  color = (0, 255, 0)     # Verde (01)
            elif val == 2:  color = (0, 0, 255)     # Rojo (10)
            else:           color = (255, 255, 255) # Blanco (11)
            
            cv2.rectangle(img, (x_start, y_start), (x_end, y_end), color, -1)
        return img

    def _transmit(self):
        if not self._send_sync_sequence():
            self.transmitting = False
            return
            
        start_time = time.monotonic()
        frames_sent = 0

        # Procesamos en bloques de 8 bytes (64 bits -> 32 celdas de 2 bits)
        for i in range(0, len(self.payload), 8):
            if not self.transmitting: break
            
            # Unimos los 8 bytes en una estructura continua de bits
            chunk = self.payload[i:i+8]
            bits = []
            for b in chunk:
                for j in range(7, -1, -1):
                    bits.append((b >> j) & 1)
            
            # Agrupamos los bits de 2 en 2 para las 32 celdas
            cells = []
            for j in range(0, 64, 2):
                cells.append((bits[j] << 1) | bits[j+1])

            img = self._draw_matrix(cells)
            cv2.imshow('Emisor 8x4 Color', img)
            frames_sent += 1

            target_time = start_time + (frames_sent * (self.frame_ms / 1000.0))
            if self._wait_until(target_time) in (27, ord('q')):
                self.transmitting = False
                break

        img = self._get_canvas()
        cv2.putText(img, "FIN 8x4", (750, 540), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 5)
        cv2.imshow('Emisor 8x4 Color', img)
        self.transmitting = False

    def _send_sync_sequence(self):
        """Secuencia: Blanco (Estabilización) -> Verde (Disparo)."""
        # 1. Blanco por 600ms (3 frames) para estabilizar brillo de la cámara
        for _ in range(3):
            img = self._draw_matrix([3] * 32)
            cv2.imshow('Emisor 8x4 Color', img)
            if self._wait_until(time.monotonic() + self.frame_ms/1000.0) != 255: 
                return False
        
        # 2. Verde por 200ms (1 frame) - Es el Trigger
        img = self._draw_matrix([1] * 32)
        cv2.imshow('Emisor 8x4 Color', img)
        if self._wait_until(time.monotonic() + self.frame_ms/1000.0) != 255: 
            return False
            
        return True

    def _wait_until(self, target_time: float) -> int:
        while time.monotonic() < target_time:
            key = cv2.waitKey(1) & 0xFF
            if key != 255: return key
        return 255

def main():
    mensaje_path = Path("mensaje.txt")
    if not mensaje_path.is_file(): mensaje_path.write_text("¡Hola mundo a color en 8x4!")
    
    # Preámbulo de 8 bytes para mantener consistencia
    payload = bytes([0x55] * 8) + mensaje_path.read_bytes()
    Transmitter8x4Color(payload).run()

if __name__ == "__main__":
    main()