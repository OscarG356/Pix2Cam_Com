import os

def calcular_ber(ruta_original, ruta_recibido):
    """
    Calcula la Bit Error Rate (BER) entre dos archivos de texto.
    
    Args:
        ruta_original: Ruta del archivo mensaje.txt
        ruta_recibido: Ruta del archivo mensaje_recibido.txt
    
    Returns:
        ber: Tasa de error de bits (BER)
    """
    
    try:
        # Leer los archivos en binario para evitar problemas de encoding
        with open(ruta_original, 'rb') as f:
            data_original = f.read()

        with open(ruta_recibido, 'rb') as f:
            data_recibido = f.read()

        # Convertir a secuencias de bytes y comparar bit a bit
        len_orig = len(data_original)
        len_rec = len(data_recibido)
        max_len_bytes = max(len_orig, len_rec)

        bits_error = 0
        total_bits = max_len_bytes * 8

        # Comparar byte a byte para las posiciones existentes
        for i in range(max_len_bytes):
            b_orig = data_original[i] if i < len_orig else 0
            b_rec = data_recibido[i] if i < len_rec else 0
            # XOR y contar bits distintos
            diff = b_orig ^ b_rec
            bits_error += bin(diff).count('1')

        ber = bits_error / total_bits if total_bits > 0 else 0
        return ber, bits_error, total_bits
    
    except FileNotFoundError as e:
        print(f"Error: Archivo no encontrado - {e}")
        return None, None, None


if __name__ == "__main__":
    ruta_original = "mensaje.txt"
    ruta_recibido = "mensaje_recibido.txt"
    
    ber, bits_error, total_bits = calcular_ber(ruta_original, ruta_recibido)
    
    if ber is not None:
        print(f"BER (Bit Error Rate): {ber:.6f}")
        print(f"Bits errados: {bits_error}")
        print(f"Total de bits: {total_bits}")
        print(f"Porcentaje de error: {ber * 100:.4f}%")
    else:
        print("No se pudo calcular el BER")
