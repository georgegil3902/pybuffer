import numpy as np
from buffer import Buffer

class QueueBuffer(Buffer):
    def __init__(self, size: int, shape: tuple | None = None, over_write: bool = False) -> None:
        """
        Initialize the QueueBuffer with a given size and shape.
        
        Parameters:
        - size (int): Number of elements the buffer can hold.
        - shape (tuple or None): The shape of each element in the buffer. If shape is None, each element is a scalar.
        - over_write (bool): If True, new elements overwrite the oldest ones when the buffer is full.
        """
        shape = () if shape is None else shape
        super().__init__(size=size, shape=shape)
        self._over_write = over_write

    def _write_operation(self, item: tuple | list | np.ndarray) -> bool:
        """
        Insert an item into the queue buffer.
        
        Parameters:
        - item (tuple, list, or np.ndarray): The item to insert into the buffer.
        
        Returns:
        - bool: True if the item was inserted successfully, False if the buffer was full and over_write is False.
        """
        # Ensure item is of correct type and shape
        if not isinstance(item, (tuple, list, np.ndarray)):
            raise ValueError(f"Item must be of type tuple, list, or np.ndarray, got {type(item)}")

        item = np.asarray(item)
        if item.shape != self._shape:
            raise ValueError(f"Item shape {item.shape} does not match buffer element shape {self._shape}")

        if self.is_full:
            if self._over_write:
                # Overwrite the oldest item by shifting everything forward
                self._buffer[:-1] = self._buffer[1:]  # Shift elements to the left
                self._buffer[-1] = item  # Place the new item at the end
            else:
                return False  # Cannot write as buffer is full and overwriting is disabled
        else:
            # Insert item at the current write pointer
            self._buffer[self._write_pointer] = item
            self._advance_write_pointer()

        return True

    def _read_operation(self) -> np.ndarray | None:
        """
        Read and remove the front item from the queue buffer.
        
        Returns:
        - np.ndarray: The item read from the buffer, or None if the buffer is empty.
        """
        if self.is_empty:
            return None  # Buffer is empty, return None
        
        item = self._buffer[0].copy()  # Get the front item
        self._buffer[:-1] = self._buffer[1:]  # Shift elements to the left
        self._write_pointer -= 1  # Adjust the write pointer
        return item

    @property
    def is_empty(self) -> bool:
        """
        Check if the queue buffer is empty.
        
        Returns:
        - bool: True if the buffer is empty, False otherwise.
        """
        return self._write_pointer == 0

    @property
    def is_full(self) -> bool:
        """
        Check if the queue buffer is full.
        
        Returns:
        - bool: True if the buffer is full, False otherwise.
        """
        return self._write_pointer >= self._size

    def _advance_write_pointer(self) -> None:
        """
        Advance the write pointer after a successful write.
        """
        if self._write_pointer < self._size:
            self._write_pointer += 1


# Example implementation
if __name__ == "__main__":
    # Example with single values
    queue = QueueBuffer(5, None, over_write=False)  # None indicates scalar values
    queue.write(1)
    queue.write(2)
    queue.write(3)
    queue.write(4)
    queue.write(5)
    print(queue.is_full)  # Expected output: True (since it should be full after 5 writes)
    print(queue.read())  # Expected output: 1
    queue.write(6)  # Should add 6 to the end of the queue
    print(queue.read())  # Expected output: 2 (since 1 was already read, 2 is next)

    # Attempt to read from an empty buffer
    while not queue.is_empty:
        print(queue.read())  # Expected output: 3, 4, 5, 6

    print(queue.read())  # Expected output: None (since the buffer is now empty)

    # Example with 2D arrays
    queue_2d = QueueBuffer(3, (2, 2), over_write=True)  # Shape = (2, 2) indicates 2D arrays
    queue_2d.write([[1, 2], [3, 4]])
    queue_2d.write([[5, 6], [7, 8]])
    queue_2d.write([[9, 10], [11, 12]])
    print(queue_2d.is_full)  # Expected output: True
    print(queue_2d.read())  # Expected output: [[1, 2], [3, 4]]
    queue_2d.write([[13, 14], [15, 16]])  # Overwrites the oldest data
    print(queue_2d.read())  # Expected output: [[5, 6], [7, 8]]
    print(queue_2d.read())  # Expected output: [[9, 10], [11, 12]]
