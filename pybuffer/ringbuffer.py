import numpy as np

from typing import Tuple, Union

from buffer import Buffer

class RingBuffer(Buffer):
    def __init__(self, size: int, shape: Tuple[int, ...], dtype: np.dtype = np.float64) -> None:
        super().__init__(size=size, shape=shape, dtype=dtype)

    def write(self, item: Union[tuple, list, np.ndarray]) -> bool:
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

    def read(self) -> np.ndarray:
        if self.is_empty:
            return None  # Buffer is empty, return None
        
        item = self._buffer[self._read_pointer]
        self._advance_read_pointer()
        return item

    def automate(self) -> None:
        pass
