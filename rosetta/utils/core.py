"""
Core utilities for Cache-to-Cache (C2C) operations.
"""

from typing import List


def sharers_to_mask(sharer_indices: List[int]) -> int:
    """
    Convert a list of sharer indices to a bitmask.
    
    Args:
        sharer_indices: List of 1-based sharer indices (e.g., [1, 2, 3])
        
    Returns:
        Bitmask integer (e.g., [1, 2] -> 3, [1, 3] -> 5, [1, 2, 3] -> 7)
    
    Example:
        >>> sharers_to_mask([1])      # 001 = 1
        1
        >>> sharers_to_mask([2])      # 010 = 2
        2
        >>> sharers_to_mask([1, 2])   # 011 = 3
        3
        >>> sharers_to_mask([1, 3])   # 101 = 5
        5
    """
    mask = 0
    for idx in sharer_indices:
        mask |= (1 << (idx - 1))
    return mask


def mask_to_sharers(mask: int) -> List[int]:
    """
    Convert a bitmask to a list of sharer indices.
    
    Args:
        mask: Bitmask integer
        
    Returns:
        List of 1-based sharer indices
    
    Example:
        >>> mask_to_sharers(1)   # 001 -> [1]
        [1]
        >>> mask_to_sharers(3)   # 011 -> [1, 2]
        [1, 2]
        >>> mask_to_sharers(5)   # 101 -> [1, 3]
        [1, 3]
        >>> mask_to_sharers(7)   # 111 -> [1, 2, 3]
        [1, 2, 3]
    """
    if mask <= 0:
        return []
    sharers = []
    idx = 1
    while mask:
        if mask & 1:
            sharers.append(idx)
        mask >>= 1
        idx += 1
    return sharers


def all_sharers_mask(num_sharers: int) -> int:
    """
    Get bitmask that selects all sharers.
    
    Args:
        num_sharers: Number of sharers
        
    Returns:
        Bitmask with all bits set (e.g., 3 sharers -> 7 = 111)
    """
    return (1 << num_sharers) - 1


def format_sharer_mask(mask: int) -> str:
    """
    Format a sharer mask as a human-readable string.
    
    Args:
        mask: Bitmask integer (-1=no projection, 0=self projection, >0=sharer bitmask)
        
    Returns:
        Formatted string like "sharers [1, 2]" or "no projection"
    """
    if mask < 0:
        return "no projection"
    if mask == 0:
        return "self projection"
    sharers = mask_to_sharers(mask)
    return f"sharers {sharers}"
