#!/usr/bin/env python3
"""
Emisor 1x1 usando OpenCV - Modulación por Color (4 colores = 2 bits/símbolo)
"""

import cv2
import numpy as np
from pathlib import Path

TX_FRAME_MS = 1016

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
        self.bits_per_symbol = 2  # Usamos 2 bits porque tenemos 4 colores
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
        # Secuencia de sincronización: Usamos ROJO puro para distinguirlo de los datos
        self._send_sync_sequence(repeats=1)

        total_bits = len(self.payload) * 8

        while self.transmitting and self.bit_index < total_bits:
            # Extraer los próximos 'bits_per_symbol' bits
            symbol_val = 0
            bits_to_read = min(self.bits_per_symbol, total_bits - self.bit_index)
            
            for i in range(bits_to_read):
                current_bit_pos = self.bit_index + i
                byte_idx = current_bit_pos // 8
                bit_in_byte = 7 - (current_bit_pos % 8)
                bit = (self.payload[byte_idx] >> bit_in_byte) & 1
                # Desplazar y añadir el bit leído
                symbol_val = (symbol_val << 1) | bit
            
            # Si al final del archivo sobran bits impares, desplazamos para alinear
            if bits_to_read < self.bits_per_symbol:
                symbol_val = symbol_val << (self.bits_per_symbol - bits_to_read)

            # Obtener el color del diccionario
            color_val = self.color_map[symbol_val]
            img = np.full((1080, 1920, 3), color_val, dtype=np.uint8)

            # Para que el texto sea visible, invertimos el color de fondo o usamos gris
            # Calcular la luminancia básica para decidir color de texto
            luminancia = 0.114*color_val[0] + 0.587*color_val[1] + 0.299*color_val[2]
            info_color = (0, 0, 0) if luminancia > 128 else (255, 255, 255)
            
            progress = f"Bits: {self.bit_index + bits_to_read}/{total_bits} | Simbolo: {bin(symbol_val)[2:].zfill(self.bits_per_symbol)}"
            cv2.putText(img, progress, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, info_color, 2)

            cv2.imshow('Emisor 1x1', img)

            self.bit_index += bits_to_read

            key = self._wait_for_frame(self.frame_ms)
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
        for i in range(repeats):
            # Rojo (Fuera de banda de datos)
            img = np.zeros((1080, 1920, 3), dtype=np.uint8)
            img[:] = (0, 0, 255) 
            cv2.putText(img, f"SYNC {i+1}/{repeats}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            cv2.imshow('Emisor 1x1', img)
            key = self._wait_for_frame(self.frame_ms)
            if key in (27, ord('q')):
                self.transmitting = False
                return

            # Negro
            img = np.zeros((1080, 1920, 3), dtype=np.uint8)
            cv2.putText(img, f"SYNC {i+1}/{repeats}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            cv2.imshow('Emisor 1x1', img)
            key = self._wait_for_frame(self.frame_ms)
            if key in (27, ord('q')):
                self.transmitting = False
                return

    def _wait_for_frame(self, ms: int | None = None) -> int:
        if ms is None:
            ms = self.frame_ms
        return cv2.waitKey(ms) & 0xFF

def main():
    mensaje_path = Path("mensaje.txt")

    if not mensaje_path.is_file():
        # Crear un archivo de prueba si no existe
        mensaje_path.write_text("Hola Mundo Colorido!")
    
    payload = mensaje_path.read_bytes()
    if not payload:
        print("Error: archivo vacío")
        return
    
    # Preámbulo de sincronización: 0x55
    sync_preamble = bytes([0x55])
    full_payload = sync_preamble + payload
    
    tx = Transmitter1x1Color(full_payload, frame_ms=TX_FRAME_MS)
    tx.run()

if __name__ == "__main__":
    main()