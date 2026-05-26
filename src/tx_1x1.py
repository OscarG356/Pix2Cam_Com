#!/usr/bin/env python3
"""
Emisor minimal 1x1 para Pix2Cam_Com.

Transmite un bit a la vez:
- NEGRO (0) = celda negra (RGB 0,0,0)
- BLANCO (1) = celda blanca (RGB 255,255,255)

Cada frame dura 1 segundo por defecto (configurable).
Presiona ENTER para iniciar, ESC para salir.

Uso:
    python3 src/tx_1x1.py archivo.txt [--frame-ms 1000]
"""

import argparse
import sys
import time
import tkinter as tk
from pathlib import Path
from PIL import Image, ImageTk


class Transmitter1x1:
    def __init__(self, root: tk.Tk, payload: bytes, frame_ms: int = 1000):
        self.root = root
        self.payload = payload
        self.frame_ms = frame_ms
        self.transmitting = False
        self.bit_index = 0
        
        self.root.title("Pix2Cam_Com - Emisor 1x1")
        self.root.geometry("600x400")
        
        # Canvas para mostrar el bit actual
        self.canvas = tk.Canvas(root, width=600, height=400, bg='gray')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Status label
        self.status = tk.Label(root, text="Listo. Presiona ENTER para iniciar.", font=('Arial', 12))
        self.status.pack(pady=10)
        
        self.root.bind('<Return>', self._start)
        self.root.bind('<Escape>', lambda e: self.root.quit())
    
    def _start(self, event=None):
        """Inicia transmisión."""
        if self.transmitting:
            return
        self.transmitting = True
        self.bit_index = 0
        self.status.config(text="Transmitiendo...")
        self._transmit_next_bit()
    
    def _transmit_next_bit(self):
        """Transmite el siguiente bit."""
        if self.bit_index >= len(self.payload) * 8:
            # Fin
            self.transmitting = False
            self.status.config(text="Transmisión completada. Presiona ENTER para otra.")
            self.canvas.create_rectangle(0, 0, 600, 400, fill='gray')
            self.canvas.create_text(300, 200, text="FIN", font=('Arial', 40), fill='white')
            return
        
        # Calcular byte y bit dentro del byte
        byte_idx = self.bit_index // 8
        bit_in_byte = 7 - (self.bit_index % 8)  # MSB primero
        
        # Extraer bit
        byte_val = self.payload[byte_idx]
        bit = (byte_val >> bit_in_byte) & 1
        
        # Mostrar en canvas
        color = 'white' if bit else 'black'
        self.canvas.delete('all')
        self.canvas.create_rectangle(0, 0, 600, 400, fill=color)
        
        # Info
        progress = f"{self.bit_index + 1}/{len(self.payload) * 8}"
        byte_info = f"Byte {byte_idx}: bit {bit_in_byte} = {bit}"
        self.status.config(text=f"{progress} | {byte_info}")
        
        self.bit_index += 1
        
        # Próximo bit en frame_ms
        self.root.after(self.frame_ms, self._transmit_next_bit)


def main():
    parser = argparse.ArgumentParser(description="Emisor 1x1 minimalista")
    parser.add_argument("file", type=Path, help="Archivo a transmitir")
    parser.add_argument("--frame-ms", type=int, default=1000, help="Duración de cada frame en ms")
    
    args = parser.parse_args()
    
    if not args.file.exists():
        print(f"Error: {args.file} no existe")
        return
    
    payload = args.file.read_bytes()
    if not payload:
        print("Error: archivo vacío")
        return
    
    print(f"Transmitiendo: {args.file}")
    print(f"Tamaño: {len(payload)} bytes = {len(payload) * 8} bits")
    print(f"Duración esperada: {len(payload) * 8 * args.frame_ms / 1000:.1f}s")
    
    root = tk.Tk()
    app = Transmitter1x1(root, payload, args.frame_ms)
    root.mainloop()


if __name__ == "__main__":
    main()
