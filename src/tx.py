#!/usr/bin/env python3
import cv2
import numpy as np
import time
from pathlib import Path

TX_FRAME_MS = 150

COLOR_MAP_4 = {
    0: (0, 0, 0),       # 00: Negro puro
    1: (0, 0, 255),     # 01: Rojo puro
    2: (0, 255, 0),     # 10: Verde puro
    3: (255, 0, 0)      # 11: Azul puro
}

class Transmitter2x2Tracked:
    def __init__(self, payload: bytes, frame_ms: int = TX_FRAME_MS):
        self.payload = payload
        self.frame_ms = frame_ms
        self.transmitting = False
        self.color_map = COLOR_MAP_4

    def _get_canvas(self):
        """Crea el lienzo base con el marco blanco de tracking siempre visible."""
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        # Marco blanco (20px de grosor)
        cv2.rectangle(img, (540, 120), (1380, 960), (255, 255, 255), -1)
        # Fondo negro interno donde irán los colores
        cv2.rectangle(img, (560, 140), (1360, 940), (0, 0, 0), -1)
        return img

    def run(self):
        cv2.namedWindow('Emisor 2x2', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.setWindowProperty('Emisor 2x2', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        img = self._get_canvas()
        cv2.putText(img, f"Matriz Dinamica: {len(self.payload)} bytes", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.putText(img, "ENTER para iniciar", (50, 1050), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.imshow('Emisor 2x2', img)
        
        while True:
            key = cv2.waitKey(100) & 0xFF
            if key in (13, ord('\r')):
                if not self.transmitting:
                    self.transmitting = True
                    self._transmit()
            if key in (27, ord('q')): break
        cv2.destroyAllWindows()

    def _transmit(self):
        self._send_sync_sequence()
        start_time = time.monotonic()
        frames_sent = 0

        for byte_idx, b in enumerate(self.payload):
            if not self.transmitting: break

            sym_tl = (b >> 6) & 3
            sym_tr = (b >> 4) & 3
            sym_bl = (b >> 2) & 3
            sym_br = b & 3

            img = self._get_canvas()
            
            # Dibujar los 4 cuadrantes dentro del marco (Centro: X=960, Y=540)
            img[140:540, 560:960] = self.color_map[sym_tl]
            img[140:540, 960:1360] = self.color_map[sym_tr]
            img[540:940, 560:960] = self.color_map[sym_bl]
            img[540:940, 960:1360] = self.color_map[sym_br]

            cv2.imshow('Emisor 2x2', img)
            frames_sent += 1

            target_time = start_time + (frames_sent * (self.frame_ms / 1000.0))
            if self._wait_until(target_time) in (27, ord('q')):
                self.transmitting = False
                break

        img = self._get_canvas()
        cv2.imshow('Emisor 2x2', img)
        self.transmitting = False

    def _send_sync_sequence(self):
        for _ in range(1):
            # SYNC CYAN (Para que no se confunda con el Rojo o Azul de los datos)
            img = self._get_canvas()
            img[140:940, 560:1360] = (255, 255, 0) # Cyan en BGR
            cv2.imshow('Emisor 2x2', img)
            if self._wait_until(time.monotonic() + self.frame_ms/1000.0) != 255: return
            
            # SYNC NEGRO
            img = self._get_canvas()
            cv2.imshow('Emisor 2x2', img)
            if self._wait_until(time.monotonic() + self.frame_ms/1000.0) != 255: return

    def _wait_until(self, target_time: float) -> int:
        while time.monotonic() < target_time:
            key = cv2.waitKey(1) & 0xFF
            if key != 255: return key
        return 255

def main():
    mensaje_path = Path("mensaje.txt")
    if not mensaje_path.is_file(): mensaje_path.write_text("hola")
    payload = bytes([0x55]) + mensaje_path.read_bytes()
    Transmitter2x2Tracked(payload).run()

if __name__ == "__main__":
    main()