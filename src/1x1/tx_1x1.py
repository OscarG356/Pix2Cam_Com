#!/usr/bin/env python3
"""
Emisor 1x1 usando OpenCV - Modulación por Color (4 colores = 2 bits/símbolo)
Con sincronización de tiempo absoluta para evitar desfases con el receptor.
"""

import cv2
import numpy as np
import time
from pathlib import Path

# Velocidad de transmisión ajustada por el usuario (534ms)
TX_FRAME_MS = 150

# Mapa de colores en formato BGR de OpenCV
# 00 -> Negro, 01 -> Azul, 10 -> Verde, 11 -> Amarillo
COLOR_MAP_4 = {
    0: (0, 0, 0),       # 00: Negro
    1: (255, 0, 0),     # 01: Azul
    2: (0, 255, 0),     # 10: Verde
    3: (0, 255, 255)    # 11: Amarillo
}

class Transmitter1x1Color:
    def __init__(self, payload: bytes, frame_ms: int = TX_FRAME_MS):
        self.payload = payload
        self.frame_ms = frame_ms
        self.transmitting = False
        self.bit_index = 0
        self.bits_per_symbol = 2  
        self.color_map = COLOR_MAP_4
    
    def run(self):
        cv2.namedWindow('Emisor 1x1', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.setWindowProperty('Emisor 1x1', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
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
        cv2.putText(img, f"Listo: {len(self.payload)} bytes = {total_bits} bits", 
                   (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        cv2.putText(img, f"Esquema: 4 Colores ({self.bits_per_symbol} bits/frame)", 
                   (100, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        cv2.putText(img, "Presiona ENTER para iniciar", 
                   (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        cv2.imshow('Emisor 1x1', img)
    
    def _transmit(self):
        # Secuencia de sincronización inicial (Rojo -> Negro)
        self._send_sync_sequence(repeats=1)

        total_bits = len(self.payload) * 8
        
        # --- CONTROL DE TIEMPO REAL ---
        # Guardamos el instante exacto en el que empiezan a transmitirse los datos reales
        start_time = time.monotonic()
        frames_sent = 0

        while self.transmitting and self.bit_index < total_bits:
            symbol_val = 0
            bits_to_read = min(self.bits_per_symbol, total_bits - self.bit_index)
            
            for i in range(bits_to_read):
                current_bit_pos = self.bit_index + i
                byte_idx = current_bit_pos // 8
                bit_in_byte = 7 - (current_bit_pos % 8)
                bit = (self.payload[byte_idx] >> bit_in_byte) & 1
                symbol_val = (symbol_val << 1) | bit
            
            if bits_to_read < self.bits_per_symbol:
                symbol_val = symbol_val << (self.bits_per_symbol - bits_to_read)

            color_val = self.color_map[symbol_val]
            img = np.full((1080, 1920, 3), color_val, dtype=np.uint8)

            luminancia = 0.114*color_val[0] + 0.587*color_val[1] + 0.299*color_val[2]
            info_color = (0, 0, 0) if luminancia > 128 else (255, 255, 255)
            
            progress = f"Bits: {self.bit_index + bits_to_read}/{total_bits} | Simbolo: {bin(symbol_val)[2:].zfill(self.bits_per_symbol)}"
            cv2.putText(img, progress, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, info_color, 2)

            cv2.imshow('Emisor 1x1', img)

            self.bit_index += bits_to_read
            frames_sent += 1

            # CALCULAR EL TIEMPO OBJETIVO MATEMÁTICO
            # frame_ms se pasa a segundos (/1000.0) y se multiplica por los frames que ya deberíamos haber mandado
            target_time = start_time + (frames_sent * (self.frame_ms / 1000.0))
            
            # Esperar de forma adaptativa hasta llegar a ese instante exacto
            key = self._wait_until(target_time)
            if key in (27, ord('q')):
                self.transmitting = False
                break

        if self.bit_index >= total_bits:
            img = np.full((1080, 1920, 3), 128, dtype=np.uint8)
            cv2.putText(img, "TRANSMISION COMPLETADA", (200, 400),
                       cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 4)
            cv2.imshow('Emisor 1x1', img)

        self.transmitting = False

    def _send_sync_sequence(self, repeats: int = 1):
        """Envía la secuencia de sincronización usando el reloj antiguo ya que es previa al flujo principal."""
        for i in range(repeats):
            # Rojo
            img = np.zeros((1080, 1920, 3), dtype=np.uint8)
            img[:] = (0, 0, 255) 
            cv2.putText(img, f"SYNC {i+1}/{repeats}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            cv2.imshow('Emisor 1x1', img)
            key = cv2.waitKey(self.frame_ms) & 0xFF
            if key in (27, ord('q')):
                self.transmitting = False
                return

            # Negro
            img = np.zeros((1080, 1920, 3), dtype=np.uint8)
            cv2.putText(img, f"SYNC {i+1}/{repeats}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            cv2.imshow('Emisor 1x1', img)
            key = cv2.waitKey(self.frame_ms) & 0xFF
            if key in (27, ord('q')):
                self.transmitting = False
                return

    def _wait_until(self, target_time: float) -> int:
        """Sloops pequeños de 1ms para mantener la ventana de OpenCV activa sin pasarse del tiempo objetivo."""
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
    
    # NOTA: Si dejas el sync_preamble activo, recuerda configurar tu 
    # receptor con EXPECTED_PAYLOAD_BYTES = len(mensaje) + 1
    sync_preamble = bytes([0x55])
    full_payload = sync_preamble + payload
    
    print(f"Transmitiendo con precisión absoluta... Tamaño total: {len(full_payload)} bytes")
    tx = Transmitter1x1Color(full_payload, frame_ms=TX_FRAME_MS)
    tx.run()

if __name__ == "__main__":
    main()