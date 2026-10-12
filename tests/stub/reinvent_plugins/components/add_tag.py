"""Minimal stand-in for REINVENT4's `reinvent_plugins.components.add_tag`.

Lives on a *separate* sys.path entry from reinvent/, so importing the real
component through it exercises the namespace-package merge REINVENT relies
on (see tests/test_reinvent_component.py). Mirrors REINVENT's own one-line
decorator: attach an attribute, return the object unchanged.
"""


def add_tag(name: str, value: str = "True"):
    """Tag a class or dataclass with `name` = `value` and return it unchanged."""

    def wrapper(obj):
        setattr(obj, name, value)
        return obj

    return wrapper
