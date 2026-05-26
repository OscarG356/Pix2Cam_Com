#!/usr/bin/env python3
"""
Emisor binario 2x4 para Pix2Cam_Com.

Simplificación robusta del emisor:
- Cada byte se representa como 8 celdas binarias (negro/blanco)
- Grilla 2x4 (2 filas x 4 columnas)
- Cada bit 1 = celda blanca, bit 0 = celda negra
- Sin niveles de gris intermedios => mucho más robusto
- Sincronización con preamble (8 bytes de referencia)
"""

from __future__ import annotations

import argparse
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List
import tkinter as tk
from PIL import Image, ImageDraw


# Configuración binaria: 8 bits por frame = 1 byte completo
GRID_ROWS = 2
GRID_COLS = 4
BITS_PER_FRAME = GRID_ROWS * GRID_COLS  # 8

# Sincronización
SYNC_PREAMBLE = bytes([0xA5, 0x5A, 0x3C, 0xC3, 0x96, 0x69, 0xF0, 0x0F])


@dataclass
class TransmitterConfig:
    cell_px: int = 160  # Píxeles por celda
    frame_ms: int = 100  # Duración del frame en ms
    quiet_zone_px: int = 48  # Borde negro alrededor
    reference_ms: int = 1000  # Fase de referencia inicial


def byte_to_bits(b: int) -> List[int]:
    """Convierte un byte a lista de 8 bits (MSB primero)."""
    return [(b >> (7 - i)) & 1 for i in range(8)]


def build_reference_frames(count: int) -> List[bytes]:
    """Crea frames de referencia con patrón aleatorio binario."""
    frames = []
    for _ in range(count):
        # Byte aleatorio
        byte_val = secrets.randbelow(256)
        frames.append(bytes([byte_val]))
    return frames


def build_packet(payload: bytes) -> bytes:
    """
    Construye paquete: [PREAMBLE][LENGTH_4B][PAYLOAD]
    
    Formato:
    - 8 bytes de preamble (sincronización)
    - 4 bytes de longitud (little-endian)
    - N bytes de payload
    """
    length_bytes = len(payload).to_bytes(4, 'little')
    return SYNC_PREAMBLE + length_bytes + payload


def create_frame_image(bits: List[int], config: TransmitterConfig) -> Image.Image:
    """
    Crea una imagen PIL con grilla 2x4 binaria.
    
    Args:
        bits: Lista de 8 bits (0=negro, 1=blanco)
        config: Configuración del emisor
    
    Returns:
        Imagen PIL con frame
    """
    cell_px = config.cell_px
    quiet = config.quiet_zone_px
    
    # Tamaño total: quiet_zone + 4 columnas + quiet_zone, similar para filas
    total_w = quiet + (GRID_COLS * cell_px) + quiet
    total_h = quiet + (GRID_ROWS * cell_px) + quiet
    
    # Crear imagen blanca
    img = Image.new('RGB', (total_w, total_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Dibujar quiet zone negro
    draw.rectangle([(0, 0), (total_w, quiet)], fill=(0, 0, 0))
    draw.rectangle([(0, total_h - quiet), (total_w, total_h)], fill=(0, 0, 0))
    draw.rectangle([(0, quiet), (quiet, total_h - quiet)], fill=(0, 0, 0))
    draw.rectangle([(total_w - quiet, quiet), (total_w, total_h - quiet)], fill=(0, 0, 0))
    
    # Dibujar celdas binarias
    bit_idx = 0
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            if bit_idx < len(bits):
                # Posición de la celda
                x0 = quiet + col * cell_px
                y0 = quiet + row * cell_px
                x1 = x0 + cell_px
                y1 = y0 + cell_px
                
                # Color: blanco (1) o negro (0)
                color = (255, 255, 255) if bits[bit_idx] else (0, 0, 0)
                draw.rectangle([(x0, y0), (x1, y1)], fill=color)
                
                # Borde gris para visualizar
                draw.rectangle([(x0, y0), (x1, y1)], outline=(128, 128, 128), width=1)
                
                bit_idx += 1
    
    return img


class TransmitterApp:
    def __init__(self, root: tk.Tk, config: TransmitterConfig):
        self.root = root
        self.config = config
        self.root.title("Pix2Cam_Com - Emisor Binario 2x4")
        
        # Canvas para mostrar frame
        self.canvas = tk.Canvas(root, width=800, height=600, bg='white')
        self.canvas.pack()
        
        # Label para estado
        self.status_label = tk.Label(root, text="Presiona ENTER para iniciar", font=('Arial', 12))
        self.status_label.pack()
        
        self.payload: bytes | None = None
        self.transmitting = False
        
        self.root.bind('<Return>', lambda e: self.start_transmission())
        self.root.bind('<Escape>', lambda e: self.root.quit())
    
    def start_transmission(self):
        """Inicia la transmisión cuando presionas ENTER."""
        if self.transmitting or self.payload is None:
            return
        
        self.transmitting = True
        self.status_label.config(text="Transmitiendo...")
        self.transmit()
    
    def transmit(self):
        """Loop de transmisión."""
        if not self.transmitting:
            return
        
        # Construir paquete
        packet = build_packet(self.payload)
        
        # Fase 1: Referencia aleatoria
        ref_frames = build_reference_frames(int(self.config.reference_ms / self.config.frame_ms))
        for ref_byte in ref_frames:
            bits = byte_to_bits(ref_byte[0])
            img = create_frame_image(bits, self.config)
            self.display_image(img)
            self.root.update()
            time.sleep(self.config.frame_ms / 1000.0)
        
        # Fase 2: Transmitir paquete
        for byte_val in packet:
            bits = byte_to_bits(byte_val)
            img = create_frame_image(bits, self.config)
            self.display_image(img)
            self.root.update()
            time.sleep(self.config.frame_ms / 1000.0)
        
        # Fin
        self.transmitting = False
        self.status_label.config(text="Transmisión completada. Presiona ENTER para otra.")
        self.display_message("FIN")
    
    def display_image(self, img: Image.Image):
        """Muestra imagen PIL en el canvas."""
        # Redimensionar si es necesario
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w > 1:
            scale = min(canvas_w / img.width, canvas_h / img.height)
            new_w = int(img.width * scale)
            new_h = int(img.height * scale)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        self.photo = tk.PhotoImage(data=img.tobytes(), width=img.width, height=img.height, format='PPM')
        self.canvas.create_image(400, 300, image=self.photo)
    
    def display_message(self, msg: str):
        """Muestra un mensaje de texto en el canvas."""
        self.canvas.delete('all')
        self.canvas.create_text(400, 300, text=msg, font=('Arial', 24))


def main():
    parser = argparse.ArgumentParser(description="Emisor binario 2x4 para Pix2Cam_Com")
    parser.add_argument("file", type=Path, help="Archivo de texto a transmitir")
    parser.add_argument("--cell-px", type=int, default=160, help="Píxeles por celda")
    parser.add_argument("--frame-ms", type=int, default=100, help="Duración del frame en ms")
    parser.add_argument("--quiet-px", type=int, default=48, help="Píxeles de borde tranquilo")
    parser.add_argument("--reference-ms", type=int, default=1000, help="Duración fase referencia")
    
    args = parser.parse_args()
    
    if not args.file.exists():
        print(f"Archivo no encontrado: {args.file}")
        return
    
    # Leer payload
    payload = args.file.read_bytes()
    if len(payload) > 10000:
        print("Archivo muy grande (máx 10000 bytes)")
        return
    
    config = TransmitterConfig(
        cell_px=args.cell_px,
        frame_ms=args.frame_ms,
        quiet_zone_px=args.quiet_px,
        reference_ms=args.reference_ms,
    )
    
    root = tk.Tk()
    app = TransmitterApp(root, config)
    app.payload = payload
    app.display_message(f"Listo para transmitir:\n{len(payload)} bytes\n\nPresiona ENTER")
    
    root.mainloop()


if __name__ == "__main__":
    main()
