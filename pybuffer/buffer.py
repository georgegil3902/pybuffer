import inspect
import numpy as np
from abc import ABC, abstractmethod
from typing import Union, Tuple

class Buffer(ABC):
    """
    An abstract base class for a buffer system, allowing subclasses to define
    the specific behavior for reading and writing to the buffer. Automations
    (such as logging or validation) can be attached to be triggered before and
    after read/write operations.
    """
    
    # Constants to define the memory layout order
    C, F, A, K = "C", "F", "A", "K"
    BEFORE_WRITE, AFTER_WRITE, BEFORE_READ, AFTER_READ = 0, 1, 2, 3

    def __init__(self, size: int, shape: Tuple[int, ...], dtype: np.dtype = np.float64, order: str = C) -> None:
        """
        Initialize the buffer with the specified size, shape, data type, and memory order.
        """
        self._size = size
        self._shape = shape
        self._dtype = dtype
        self._order = order

        self._isfull = False
        self._isempty = True

        self._write_pointer = 0
        self._read_pointer = 0

        self._buffer = np.zeros(shape=(self._size,) + self._shape, dtype=dtype, order=self._order)
        self._automation_results = {}

        # Dictionary to store automation functions for each event
        self.automations = {
            self.BEFORE_WRITE: [],
            self.AFTER_WRITE: [],
            self.BEFORE_READ: [],
            self.AFTER_READ: [],
        }

    def write(self, item: Union[tuple, list, np.ndarray]) -> bool:
        """
        Write operation with automation triggers, but actual implementation is delegated to subclass.
        """
        self._trigger_automations(self.BEFORE_WRITE)
        success = self._write_operation(item)  # Subclass defines how the actual write happens
        self._trigger_automations(self.AFTER_WRITE)
        return success

    def read(self, with_automation_results: bool = False) -> np.ndarray:
        """
        Read operation with automation triggers, but actual implementation is delegated to subclass.
        """
        self._trigger_automations(self.BEFORE_READ)
        item = self._read_operation(with_automation_results)  # Subclass defines how the actual read happens
        self._trigger_automations(self.AFTER_READ)
        return item

    @abstractmethod
    def _write_operation(self, item: Union[tuple, list, np.ndarray]) -> bool:
        """
        Subclass must implement this method to define how writing happens.
        """
        pass

    @abstractmethod
    def _read_operation(self, with_automation_results: bool = False) -> np.ndarray:
        """
        Subclass must implement this method to define how reading happens.
        """
        pass

    @abstractmethod
    def get_buffer(self, with_automation_results: bool = False):
        """
        Function that returns the entire buffer
        """
        pass
    
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

    def _trigger_automations(self, event: int) -> None:
        """
        Trigger automations for a given event (BEFORE/AFTER READ/WRITE).
        """
        for condition, function, result_column in self.automations[event]:
            if condition is None or condition(self):
                result = function(self)
                if result_column is not None:
                    if event == self.AFTER_WRITE:
                        self._automation_results[result_column][self._write_pointer - 1] = result
                    elif event== self.AFTER_READ:
                        self._automation_results[result_column][self._read_pointer - 1] = result
                    elif event== self.BEFORE_WRITE:
                        self._automation_results[result_column][self._write_pointer] = result
                    elif event== self.BEFORE_READ:
                        self._automation_results[result_column][self._read_pointer] = result



    def _check_function_signature(self, function: callable) -> bool:
        """
        Check if the function signature is valid (i.e., it accepts exactly one argument for 'self').

        Parameters:
        - function (callable): The function to check.

        Returns:
        - bool: True if the function accepts one argument, False otherwise.
        """
        try:
            # Get the signature of the function
            sig = inspect.signature(function)
            parameters = sig.parameters

            # Check if the function has exactly one parameter (which should be 'self')
            if len(parameters) == 1:
                return True
            else:
                return False
        except ValueError:
            # In case the function signature cannot be inspected (e.g., if it's a built-in function)
            return False

    def add_automation(self, function: callable, when: int, condition: callable = None, store_result_as: str = None):
        """
        Add an automation function that will be triggered on a specific event.

        Parameters:
        - function (callable): The function to call when the event occurs.
        - when (int): The event type (BEFORE_WRITE, AFTER_WRITE, BEFORE_READ, AFTER_READ).
        - condition (callable): A condition function that must return True for the automation to trigger.

        Raises:
        - ValueError: If the 'when' parameter is not valid or the function signature is invalid.
        """
        if when not in [self.BEFORE_READ, self.BEFORE_WRITE, self.AFTER_READ, self.AFTER_WRITE]:
            raise ValueError("Invalid event type for automation.")
        
        # Check if the function signature is valid (i.e., accepts one argument for 'self')
        if not self._check_function_signature(function):
            raise ValueError("The automation function must accept exactly one argument (self / Buffer instance).")
        
        # Add the automation if the function signature is valid
        if function not in self.automations[when]:
            self.automations[when].append((condition, function, store_result_as))
            if store_result_as is not None:
                self._add_automation_results_column(store_result_as)

    def _add_automation_results_column(self, name):
        _new_column = np.zeros(shape=(self._size,1), dtype=self._dtype, order=self._order)
        self._automation_results[name] = _new_column

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
        return self._isempty

    @property
    def is_full(self) -> bool:
        """
        Check if the buffer is full.

        Returns:
        - bool: True if the buffer is full, False otherwise.
        """
        # The buffer is full if the write pointer is at the same position as the read pointer
        # and the element at the read pointer is not the default zero (indicating that it has been written to).
        return self._isfull