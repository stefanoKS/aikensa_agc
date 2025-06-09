import yaml
import random
from typing import List, Dict

def list_to_16bit_int(bit_list: List[int]) -> int:
    if len(bit_list) > 16:
        raise ValueError("Input list must be 16 bits or fewer")
    
    # Pad to 16 bits by adding leading zeros
    full_bits = [0] * (16 - len(bit_list)) + bit_list

    # Convert to binary string and then to int
    bit_str = ''.join(str(b) for b in full_bits)
    return int(bit_str, 2)

def load_register_map(yaml_path: str) -> Dict[str, int]:
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    # Invert the mapping: {label: address}
    return {v: int(k) for k, v in data.items()}

def invert_16bit_int(value: int) -> int:
    """
    Inverts all bits of a 16-bit integer.
    Example: 0b0000000000001010 -> 0b1111111111110101
    """
    if not 0 <= value <= 0xFFFF:
        raise ValueError("Input must be a 16-bit unsigned integer (0 to 65535)")
    return value ^ 0xFFFF

def random_list(length: int) -> List[int]:
    """
    Generates a random list of 0s and 1s of specified length.
    """
    if length < 0:
        raise ValueError("Length must be a non-negative integer")
    return [random.randint(0, 1) for _ in range(length)]
