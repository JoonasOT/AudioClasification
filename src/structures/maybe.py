from __future__ import annotations

import copy
import traceback
import sys

from typing import TypeVar, Generic, Callable, Union


T = TypeVar('T')
V = TypeVar('V')


class Maybe(Generic[T]):
    """
        Wraps given value to which it can transform to different values and can run given
        functions with the wrapped value.
    """

    class Empty:
        """
            Basically an equivalent class to Optional.Empty, but just to the Maybe class. Can be supplied with a
            reason why we got an empty.
        """

        def __init__(self, reason: str = ""):
            self.reason = reason

        def __str__(self) -> str:
            RED = '\033[1;4;91m'
            NORMAL = '\033[0;0m'
            HAS_REASON = self.reason == ''
            return f"{RED}{'This is empty' if HAS_REASON else 'Empty since'}" \
                   f"{NORMAL}{'!' if HAS_REASON else ': ' + self.reason}"

    def __init__(self, v: T):
        """
        Create wrapper 'Maybe[T]' for a given value of type T.
        :param v: Value to be wrapped
        """
        self.v: T = v

    def apply(self, f: Callable[[T], V], unpack: bool = False) -> Union[V, Maybe.Empty]:
        """Apply the wrapped variable to a given function and return the value or an Empty.

        :param f: The function to be applied
        :param unpack: Boolean flag for unwrapping tuples before applying them to the given function
        :return: result of the wrapped value given to f
        """
        value = copy.deepcopy(self.v)
        func_calls = [lambda: f(value), lambda: f(*value)]
        if bool(self):
            try:
                return func_calls[int(unpack)]()
            except Exception as exp:
                stack_trace = traceback.format_stack()
                stack_trace.reverse()
                return Maybe.Empty(str(exp) + f". Traceback:\n{''.join(stack_trace[1:])}")

    def transform(self, f: Callable[[T], V], unpack: bool = False) -> Maybe[Union[V, Maybe.Empty]]:
        """Apply the given function and wrap the value in a new Maybe

        :param f: Function to be applied
        :param unpack: Boolean flag for if we need to unpack tuples before applying
        :return: A new maybe wrapping the result of the application of f
        """
        return Maybe(self.apply(f, unpack))

    def run(self, f: Callable[[T], V], unpack: bool = False) -> Maybe[T]:
        """Run function f and if it results in an exception print the info to console

        :param f: Function to be run on the wrapped value
        :param unpack: Boolean flag for if we need to unpack tuples before applying
        :return: self
        """
        val: V = self.apply(f, unpack)
        if isinstance(val, Maybe.Empty):
            print(val, file=sys.stderr)
        return self

    def get(self) -> T:
        """Get the wrapped value

        :return: The wrapped value
        """
        return self.v

    def __bool__(self) -> bool:
        """Is the wrapped value an empty

        :return: True if the wrapped value is an empty
        """
        return not isinstance(self.v, Maybe.Empty)

    def __str__(self) -> str:
        """Convert to string

        :return: String representing the structure of the Maybe
        """
        return "Maybe[ value:" + type(self.v).__name__ + " = " + str(self.v) + " ]"
