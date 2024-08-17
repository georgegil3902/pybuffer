import numpy as np
from abc import ABC, abstractmethod
from typing import Union, Tuple

class Buffer(ABC):
    # Constants to define the memory layout order
    C, F, A, K = "C", "F", "A", "K"

    def __init__(self, size: int, shape: Tuple[int, ...], dtype: np.dtype = np.float64, order: str = C) -> None:
        """
        Initialize the buffer with a specific size, shape, data type, and memory order.

        Parameters:
        - size (int): Number of elements the buffer can hold.
        - shape (tuple): Shape of each element in the buffer (e.g., a 1D array with shape (3,) or 2D array with shape (3, 3)).
        - dtype (np.dtype): Data type of the elements (default: np.float64).
        - order (str): Memory layout order, e.g., 'C' (row-major), 'F' (column-major), 'A' (Any), or 'K' (keep layout) (default: 'C').
        """
        self._size = size
        self._shape = shape
        self._dtype = dtype
        self._order = order

        self._write_pointer = 0  # Pointer indicating the next position to write
        self._read_pointer = 0   # Pointer indicating the next position to read

        # Create the buffer as a NumPy array with the specified shape and dtype
        self._buffer = np.zeros(shape=(self._size,) + self._shape, dtype=dtype, order=self._order)

    def write(self, item: Union[tuple, list, np.ndarray]) -> bool:
        """
        Write an item to the buffer at the current write pointer position and advance the pointer.

        Parameters:
        - item (Union[tuple, list, np.ndarray]): The item to write to the buffer.

        Returns:
        - bool: True if the write operation is successful.

        Raises:
        - ValueError: If the item's shape does not match the buffer's shape.
        - TypeError: If the item's type is not compatible with the buffer.
        """
        # Check the shape and type of the item before writing
        if self._shape:
            if isinstance(item, (list, tuple)):
                item = np.array(item, dtype=self._dtype)
            if isinstance(item, np.ndarray):
                if item.shape != self._shape:
                    raise ValueError(f"Expected data of shape {self._shape}, but got {item.shape}.")
            else:
                raise TypeError("Expected input to be of type tuple, list, or ndarray.")
        else:
            if not isinstance(item, (int, float, str)):
                raise TypeError("Expected input to be of type int, float, or str.")

        # Write the item to the buffer and advance the write pointer
        self._buffer[self._write_pointer] = item
        self._advance_write_pointer()
        return True

    def read(self) -> np.ndarray:
        """
        Read an item from the buffer at the current read pointer position and advance the pointer.

        Returns:
        - np.ndarray: The item read from the buffer.

        Raises:
        - RuntimeError: If the buffer is empty and there's nothing to read.
        """
        if self.is_empty:
            raise RuntimeError("Buffer is empty, cannot read.")
        item = self._buffer[self._read_pointer]
        self._advance_read_pointer()
        return item

    def _advance_write_pointer(self) -> None:
        """
        Advance the write pointer to the next position in the buffer, wrapping around if necessary.
        This function is typically called after a successful write operation.
        """
        self._write_pointer = (self._write_pointer + 1) % self._size

    def _advance_read_pointer(self) -> None:
        """
        Advance the read pointer to the next position in the buffer, wrapping around if necessary.
        This function is typically called after a successful read operation.
        """
        self._read_pointer = (self._read_pointer + 1) % self._size

    def _retract_write_pointer(self) -> None:
        """
        Retract the write pointer to the previous position in the buffer, wrapping around if necessary.
        This function is the opposite of advancing the write pointer.
        """
        self._write_pointer = (self._write_pointer - 1) % self._size

    def _retract_read_pointer(self) -> None:
        """
        Retract the read pointer to the previous position in the buffer, wrapping around if necessary.
        This function is the opposite of advancing the read pointer.
        """
        self._read_pointer = (self._read_pointer - 1) % self._size

    @property
    def size(self) -> int:
        """
        Get the size of the buffer (i.e., the number of elements it can hold).

        Returns:
        - int: The size of the buffer.
        """
        return self._size

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Get the shape of each element in the buffer.

        Returns:
        - tuple: The shape of each element in the buffer.
        """
        return self._shape

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

    @abstractmethod
    def automate(self) -> None:
        """
        Abstract method to be implemented by subclasses for automation tasks.

        This method is intended to update or manage the buffer's behavior or contents automatically.
        """
        pass
