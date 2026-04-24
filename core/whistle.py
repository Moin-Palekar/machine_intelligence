"""
Whistle — external surprise injection.

[ key  →  HIGH surprise  (press when dog is on red circle)
] key  →  LOW surprise   (press when dog is on green circle)

This is the "whistle" signal. It bypasses normal prediction error
and floods the brain with a valenced surprise state directly.
The brain still runs its normal forward/update pass — only the
surprise signal used for Hebbian scaling and logging is overridden.
"""

from pynput import keyboard

# injection values
HIGH_SURPRISE = 2.0    # strong bad signal — red circle
LOW_SURPRISE  = 0.01   # near-zero good signal — green circle

_injection = None      # None | ('high', value) | ('low', value)


def _on_press(key):
    global _injection
    try:
        ch = key.char
    except AttributeError:
        return

    if ch == '[':
        _injection = ('high', HIGH_SURPRISE)
    elif ch == ']':
        _injection = ('low', LOW_SURPRISE)


def _on_release(key):
    global _injection
    try:
        ch = key.char
    except AttributeError:
        return

    if ch in ('[', ']'):
        _injection = None


# start listener as daemon thread
_listener = keyboard.Listener(
    on_press=_on_press,
    on_release=_on_release
)
_listener.daemon = True
_listener.start()


def get_injection():
    """
    Returns current injection state:
      ('high', 2.0)   if [ is held
      ('low',  0.01)  if ] is held
      None            if neither held
    """
    return _injection