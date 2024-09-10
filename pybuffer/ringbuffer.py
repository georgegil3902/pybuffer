import numpy as np

from typing import Tuple, Union

from buffer import Buffer

class RingBuffer(Buffer):
    def __init__(self, size: int, shape: Tuple[int, ...], dtype: np.dtype = np.float64) -> None:
        super().__init__(size=size, shape=shape, dtype=dtype)

    def _write_operation(self, item: Union[tuple, list, np.ndarray]) -> bool:
        if not isinstance(item, (tuple, list, np.ndarray)):
            raise ValueError(f"Item must be of type tuple, list, or np.ndarray, got {type(item)}")

        item = np.asarray(item)
        if item.shape != self._shape:
            raise ValueError(f"Item shape {item.shape} does not match buffer element shape {self._shape}")

        self._buffer[self._write_pointer] = item
        if self.is_full:
            self._advance_read_pointer()
        self._advance_write_pointer()

        return True

    def _read_operation(self) -> np.ndarray:
        if self.is_empty:
            return None  # Buffer is empty, return None
        
        item = self._buffer[self._read_pointer]
        self._advance_read_pointer()
        return item

    @property
    def is_empty(self) -> bool:
        """
        Check if the buffer is empty.

        Returns:
        - bool: True if the buffer is empty, False otherwise.
        """
        return self._write_pointer == self._read_pointer and not self.is_full

    @property
    def is_full(self) -> bool:
        """
        Check if the buffer is full.

        Returns:
        - bool: True if the buffer is full, False otherwise.
        """
        # The buffer is full if the write pointer is at the same position as the read pointer
        # and the element at the read pointer is not the default zero (indicating that it has been written to).
        return self._write_pointer == self._read_pointer and self._buffer[self._read_pointer].any()
