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
        # Leer los archivos
        with open(ruta_original, 'r') as f:
            mensaje_original = f.read()
        
        with open(ruta_recibido, 'r') as f:
            mensaje_recibido = f.read()
        
        # Convertir a bits
        bits_original = ''.join(format(ord(c), '08b') for c in mensaje_original)
        bits_recibido = ''.join(format(ord(c), '08b') for c in mensaje_recibido)
        
        # Asegurar que tienen la misma longitud
        max_len = max(len(bits_original), len(bits_recibido))
        bits_original = bits_original.ljust(max_len, '0')
        bits_recibido = bits_recibido.ljust(max_len, '0')
        
        # Contar bits errados
        bits_error = sum(b1 != b2 for b1, b2 in zip(bits_original, bits_recibido))
        
        # Calcular BER
        ber = bits_error / max_len if max_len > 0 else 0
        
        return ber, bits_error, max_len
    
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
