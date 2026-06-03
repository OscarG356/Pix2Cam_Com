#!/usr/bin/env python3
"""
Emisor minimal 1x1 usando OpenCV.

Transmite un bit a la vez:
- NEGRO (0) = pantalla negra
- BLANCO (1) = pantalla blanca

Uso:
    python3 src/tx_1x1.py archivo.txt

Controles:
    ENTER = iniciar transmisión
    ESC/q = salir
"""

import cv2
import numpy as np
from pathlib import Path


# Cambia este valor para ajustar la velocidad de transmisión.
# Menor = más rápido. Mayor = más lento.
TX_FRAME_MS = 500


class Transmitter1x1:
    def __init__(self, payload: bytes, frame_ms: int = TX_FRAME_MS):
        self.payload = payload
        self.frame_ms = frame_ms
        self.transmitting = False
        self.bit_index = 0
    
    def run(self):
        """Loop principal."""
        # Crear ventana
        cv2.namedWindow('Emisor 1x1', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.setWindowProperty('Emisor 1x1', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        # Mostrar pantalla inicial
        self._show_ready()
        
        while True:
            key = cv2.waitKey(100) & 0xFF
            
            if key == ord('\r') or key == 13:  # ENTER
                if not self.transmitting:
                    self.transmitting = True
                    self.bit_index = 0
                    self._transmit()
            
            if key in (27, ord('q')):  # ESC o q
                break
        
        cv2.destroyAllWindows()
    
    def _show_ready(self):
        """Muestra pantalla de espera."""
        img = np.full((1080, 1920, 3), 128, dtype=np.uint8)
        cv2.putText(img, f"Listo: {len(self.payload)} bytes = {len(self.payload)*8} bits", 
                   (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        cv2.putText(img, "Presiona ENTER para iniciar", 
                   (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        cv2.imshow('Emisor 1x1', img)
    
    def _transmit(self):
        """Transmite todos los bits."""
        # Primero enviar secuencia de sincronización visual: verde -> negro (una vez)
        self._send_sync_sequence(repeats=1)

        # Luego transmitir el payload bit a bit
        while self.transmitting and self.bit_index < len(self.payload) * 8:
            byte_idx = self.bit_index // 8
            bit_in_byte = 7 - (self.bit_index % 8)
            byte_val = self.payload[byte_idx]
            bit = (byte_val >> bit_in_byte) & 1

            color_val = 255 if bit else 0
            img = np.full((1080, 1920, 3), color_val, dtype=np.uint8)

            info_color = (0, 0, 0) if bit else (255, 255, 255)
            progress = f"{self.bit_index + 1}/{len(self.payload) * 8}"
            cv2.putText(img, progress, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, info_color, 2)

            cv2.imshow('Emisor 1x1', img)

            self.bit_index += 1

            key = self._wait_for_frame(self.frame_ms)
            if key in (27, ord('q')):
                self.transmitting = False
                break

        if self.bit_index >= len(self.payload) * 8:
            img = np.full((1080, 1920, 3), 128, dtype=np.uint8)
            cv2.putText(img, "TRANSMISION COMPLETADA", (200, 400),
                       cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 4)
            cv2.putText(img, "Presiona ENTER para otra", (200, 550),
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            cv2.imshow('Emisor 1x1', img)

        self.transmitting = False

    def _send_sync_sequence(self, repeats: int = 1):
        """Muestra `repeats` veces: verde luego negro (cada uno `frame_ms` ms)."""
        for i in range(repeats):
            # Verde
            img = np.zeros((1080, 1920, 3), dtype=np.uint8)
            img[:] = (0, 255, 0)
            cv2.putText(img, f"SYNC {i+1}/{repeats}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
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
        """Espera `ms` milisegundos y devuelve la tecla pulsada, o 255 si no hubo tecla."""
        if ms is None:
            ms = self.frame_ms
        return cv2.waitKey(ms) & 0xFF


def main():
    mensaje_path = Path("mensaje.txt")

    if not mensaje_path.is_file():
        print(f"Error: mensaje.txt no existe")
        return
    
    payload = mensaje_path.read_bytes()
    if not payload:
        print("Error: archivo vacío")
        return
    
    # Agregar preámbulo de sincronización: 0x55 = 01010101 (patrón alternado perfecto)
    sync_preamble = bytes([0x55])  # 1 byte de patrón alternado = 7 transiciones
    full_payload = sync_preamble + payload
    
    print(f"Transmitiendo: mensaje.txt")
    print(f"Tamaño payload: {len(payload)} bytes")
    print(f"Con preamble: {len(full_payload)} bytes = {len(full_payload) * 8} bits")
    print(f"Preamble: 0x55 = 01010101 (7 transiciones)")
    print(f"Duración por frame: {TX_FRAME_MS} ms")
    
    tx = Transmitter1x1(full_payload, frame_ms=TX_FRAME_MS)
    tx.run()


if __name__ == "__main__":
    main()
