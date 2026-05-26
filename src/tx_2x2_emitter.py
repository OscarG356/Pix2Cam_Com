#!/usr/bin/env python3
"""
Emisor piloto para un canal pantalla->camara con una matriz 2x2.

Idea general:
- Se lee un archivo de texto en UTF-8.
- Cada byte se parte en 4 grupos de 2 bits.
- Cada grupo de 2 bits se representa con un nivel de gris.
- Los 4 niveles se dibujan en una grilla 2x2 y se muestran como una secuencia
  de cuadros en una ventana de Tkinter.

Este archivo es un piloto deliberadamente simple. Sirve como base para luego
agregar:
- preambulo y sincronizacion de trama,
- mas celdas por cuadro,
- mas niveles por celda,
- codificacion de canal,
- markers fiduciales y control de brillo.

Puntos rapidos para subir la tasa de transmision:
1. Reducir FRAME_MS.
2. Aumentar BITS_PER_CELL.
3. Aumentar GRID_SIZE (por ejemplo 3x3 o 4x4).
4. Quitar o minimizar el quiet zone si el receptor ya es robusto.
5. Usar codificacion de canal compacta y un preambulo mas corto.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence
import tkinter as tk


# Nivel actual del piloto:
# - 2 bits por celda => 4 niveles de gris.
# - Con una grilla 2x2 se transmiten 8 bits por cuadro, o sea 1 byte.
# Si quieres mas tasa, sube BITS_PER_CELL o GRID_SIZE.
BITS_PER_CELL = 2
GRID_SIZE = 2
CELL_LEVELS = (0, 85, 170, 255)

# Preámbulo compartido con el receptor.
# Debe ser una secuencia poco probable en texto normal para que el receptor
# pueda reconocer el inicio de la trama de forma robusta.
SYNC_PREAMBLE = bytes([0xA5, 0x5A, 0x3C, 0xC3, 0x96, 0x69, 0xF0, 0x0F])


@dataclass(frozen=True)
class FramePayload:
    """Un cuadro transmitido por el emisor."""

    index: int
    byte_value: int
    symbol_values: Sequence[int]


def read_text_bytes(path: Path) -> bytes:
    """Lee el archivo de texto como bytes UTF-8."""
    return path.read_text(encoding="utf-8").encode("utf-8")


def chunk_bytes(data: bytes, chunk_size: int) -> List[bytes]:
    """Parte los bytes en bloques consecutivos."""
    return [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]


def byte_to_symbols(value: int) -> List[int]:
    """
    Convierte un byte en 4 simbolos de 2 bits.

    Orden:
    - bits 7..6
    - bits 5..4
    - bits 3..2
    - bits 1..0

    La salida esta pensada para llenar una grilla 2x2 en orden fila-major.
    """
    mask = (1 << BITS_PER_CELL) - 1
    shift_values = (6, 4, 2, 0)
    return [(value >> shift) & mask for shift in shift_values]


def symbols_to_levels(symbols: Sequence[int]) -> List[int]:
    """Mapea simbolos discretos a niveles de gris visibles en pantalla."""
    return [CELL_LEVELS[s] for s in symbols]


def encode_payload(data: bytes) -> List[FramePayload]:
    """
    Codifica el archivo en cuadros de 1 byte.

    Cada byte se dibuja como una matriz 2x2:

        [s0, s1]
        [s2, s3]

    donde cada si toma uno de 4 niveles de gris.
    """
    frames: List[FramePayload] = []
    for index, byte_value in enumerate(data):
        symbols = byte_to_symbols(byte_value)
        frames.append(
            FramePayload(
                index=index,
                byte_value=byte_value,
                symbol_values=symbols_to_levels(symbols),
            )
        )
    return frames


def build_packet(data: bytes) -> bytes:
    """
    Construye el paquete transmitido.

    Estructura:
    [PREAMBLE][LENGTH_4_BYTES_BIG_ENDIAN][PAYLOAD]

    El campo LENGTH permite que el receptor sepa exactamente cuántos bytes leer
    después de sincronizarse con el preámbulo.
    """
    length_bytes = len(data).to_bytes(4, byteorder="big", signed=False)
    return SYNC_PREAMBLE + length_bytes + data


class TransmitterApp:
    """Ventana que presenta los cuadros a la camara."""

    def __init__(
        self,
        frames: Sequence[FramePayload],
        *,
        frame_ms: int,
        cell_px: int,
        quiet_zone_px: int,
        hold_end_ms: int,
        fullscreen: bool,
    ) -> None:
        self.frames = list(frames)
        self.frame_ms = frame_ms
        self.cell_px = cell_px
        self.quiet_zone_px = quiet_zone_px
        self.hold_end_ms = hold_end_ms
        self.fullscreen = fullscreen
        self.started = False

        self.root = tk.Tk()
        self.root.configure(background="black")
        self.root.title("Emisor 2x2")
        self.root.bind("<Escape>", lambda _event: self.root.destroy())
        self.root.bind("<Return>", lambda _event: self.start())
        if self.fullscreen:
            self.root.attributes("-fullscreen", True)

        # Tamaño del lienzo de transmision. El marco exterior negro funciona como
        # zona de silencio visual (quiet zone) para facilitar futuras detecciones.
        self.canvas_side = quiet_zone_px * 2 + cell_px * GRID_SIZE
        self.canvas = tk.Canvas(
            self.root,
            width=self.canvas_side,
            height=self.canvas_side,
            highlightthickness=0,
            bg="black",
        )
        self.canvas.pack(expand=True)

        self.current_frame = 0
        self.cell_coords = self._build_cell_coords()
        self._render_idle_screen()

    def _build_cell_coords(self) -> List[tuple[int, int, int, int]]:
        coords: List[tuple[int, int, int, int]] = []
        start = self.quiet_zone_px
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                x0 = start + col * self.cell_px
                y0 = start + row * self.cell_px
                x1 = x0 + self.cell_px
                y1 = y0 + self.cell_px
                coords.append((x0, y0, x1, y1))
        return coords

    def _draw_frame(self, frame: FramePayload) -> None:
        self.canvas.delete("all")
        for coords, level in zip(self.cell_coords, frame.symbol_values):
            x0, y0, x1, y1 = coords
            self.canvas.create_rectangle(
                x0,
                y0,
                x1,
                y1,
                fill=f"#{level:02x}{level:02x}{level:02x}",
                outline="#202020",
                width=2,
            )

        # Borde externo del area de transmision.
        self.canvas.create_rectangle(
            self.quiet_zone_px,
            self.quiet_zone_px,
            self.canvas_side - self.quiet_zone_px,
            self.canvas_side - self.quiet_zone_px,
            outline="#404040",
            width=2,
        )
        self.root.title(f"Emisor 2x2 | frame {frame.index + 1}/{len(self.frames)} | byte=0x{frame.byte_value:02X}")

    def _render_idle_screen(self) -> None:
        self.canvas.delete("all")
        self.canvas.create_text(
            self.canvas_side // 2,
            self.canvas_side // 2,
            fill="white",
            text="Listo para transmitir\nEnter para empezar\n(Esc para salir)",
            font=("TkDefaultFont", 16),
            justify="center",
        )

    def start(self) -> None:
        """Inicia la transmisión cuando el usuario presiona Enter."""
        if self.started:
            return
        self.started = True
        self.root.title("Emisor 2x2 | transmitiendo...")
        self.root.after(0, self._step)

    def _step(self) -> None:
        if not self.started:
            return
        if self.current_frame >= len(self.frames):
            self._render_idle_screen()
            self.root.after(self.hold_end_ms, self.root.destroy)
            return

        frame = self.frames[self.current_frame]
        self._draw_frame(frame)
        self.current_frame += 1
        self.root.after(self.frame_ms, self._step)

    def run(self) -> None:
        self.root.after(0, self._step)
        self.root.mainloop()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Emisor piloto 2x2 para texto en pantalla."
    )
    parser.add_argument(
        "text_file",
        type=Path,
        help="Archivo de texto UTF-8 a transmitir.",
    )
    parser.add_argument(
        "--frame-ms",
        type=int,
        default=100,
        help="Duracion de cada cuadro en milisegundos. Baja este valor para subir la tasa.",
    )
    parser.add_argument(
        "--cell-px",
        type=int,
        default=320,
        help="Tamanio en pixeles de cada celda. Si lo subes, mejora la robustez pero baja la tasa.",
    )
    parser.add_argument(
        "--quiet-zone-px",
        type=int,
        default=96,
        help="Margen negro alrededor de la grilla. Reducelo si luego el receptor no lo necesita.",
    )
    parser.add_argument(
        "--hold-end-ms",
        type=int,
        default=1500,
        help="Tiempo que se mantiene la pantalla al terminar la transmision.",
    )
    parser.add_argument(
        "--fullscreen",
        action="store_true",
        help="Mostrar en pantalla completa.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if not args.text_file.exists():
        raise FileNotFoundError(f"No existe el archivo: {args.text_file}")
    if not args.text_file.is_file():
        raise ValueError(f"La ruta no es un archivo: {args.text_file}")
    if args.frame_ms <= 0:
        raise ValueError("--frame-ms debe ser mayor que cero")
    if args.cell_px <= 0:
        raise ValueError("--cell-px debe ser mayor que cero")
    if args.quiet_zone_px < 0:
        raise ValueError("--quiet-zone-px no puede ser negativo")

    data = read_text_bytes(args.text_file)
    packet = build_packet(data)
    frames = encode_payload(packet)

    # Si el archivo esta vacio, igual mostramos una pantalla estable para que el
    # usuario sepa que el programa esta vivo.
    if not frames:
        frames = [
            FramePayload(index=0, byte_value=0x00, symbol_values=symbols_to_levels([0, 0, 0, 0]))
        ]

    app = TransmitterApp(
        frames,
        frame_ms=args.frame_ms,
        cell_px=args.cell_px,
        quiet_zone_px=args.quiet_zone_px,
        hold_end_ms=args.hold_end_ms,
        fullscreen=args.fullscreen,
    )
    app.run()


if __name__ == "__main__":
    main()
