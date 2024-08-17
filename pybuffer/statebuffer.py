from typing import Union, Optional
import numpy as np
from buffer import Buffer

class StateBuffer(Buffer):
    def __init__(self, size: int, shape: tuple, namespace: Optional[tuple[str]] = None) -> None:
        """
        Initialize the StateBuffer with a fixed size, shape, and an optional namespace.

        Parameters:
        - size (int): Number of elements the buffer can hold.
        - shape (tuple): Shape of each element in the buffer.
        - namespace (Optional[tuple[str]]): Optional tuple of string keys for indexed access.
                                           If provided, its length must match the size of the buffer.

        Raises:
        - ValueError: If the namespace length does not match the size.
        """
        if namespace is not None and len(namespace) != size:
            raise ValueError("Namespace length must match the buffer size.")
        
        self._namespace = namespace
        super().__init__(size=size, shape=shape)

    def write(self, item: np.ndarray, position: Union[int, str]) -> bool:
        """
        Write an item to the buffer at the specified position.

        Parameters:
        - item (np.ndarray): The item to write to the buffer.
        - position (Union[int, str]): The position in the buffer to write to.
                                      If an int, writes by index; if a str, writes by key.

        Returns:
        - bool: True if the write operation is successful.

        Raises:
        - ValueError: If the position type does not match the expected type or if the key is not found.
        - IndexError: If the position index is out of bounds.
        """
        if isinstance(position, str) and self._namespace is not None:
            try:
                idx = self._namespace.index(position)
                self._buffer[idx] = item
            except ValueError:
                raise ValueError(f"Position key '{position}' not found in namespace.")
        elif isinstance(position, int) and self._namespace is None:
            if 0 <= position < self._size:
                self._buffer[position] = item
            else:
                raise IndexError(f"Position index {position} is out of bounds for buffer of size {self._size}.")
        else:
            expected_type = str if self._namespace is not None else int
            raise ValueError(f"Expected position indexing of type {expected_type} but got {type(position)}.")
        
        return True
    
    def read(self, position: Union[int, str]) -> np.ndarray:
        """
        Read an item from the buffer at the specified position.

        Parameters:
        - position (Union[int, str]): The position in the buffer to read from.
                                      If an int, reads by index; if a str, reads by key.

        Returns:
        - np.ndarray: The item read from the buffer.

        Raises:
        - ValueError: If the position type does not match the expected type or if the key is not found.
        - IndexError: If the position index is out of bounds.
        """
        if isinstance(position, str) and self._namespace is not None:
            try:
                idx = self._namespace.index(position)
                return self._buffer[idx]
            except ValueError:
                raise ValueError(f"Position key '{position}' not found in namespace.")
        elif isinstance(position, int) and self._namespace is None:
            if 0 <= position < self._size:
                return self._buffer[position]
            else:
                raise IndexError(f"Position index {position} is out of bounds for buffer of size {self._size}.")
        else:
            expected_type = str if self._namespace is not None else int
            raise ValueError(f"Expected position indexing of type {expected_type} but got {type(position)}.")

    def automate(self) -> None:
        """
        This method is not implemented in this class
        Calling this function does nothing
        """
        pass
