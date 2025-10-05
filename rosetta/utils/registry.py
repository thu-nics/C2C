"""
Unified registry utilities and simple JSON-based save/load helpers.

This module provides:
- create_registry: factory to create (registry dict, register decorator, get_class)
- capture_init_args: decorator to record __init__ kwargs on instances as _init_args
- save_object / load_object: serialize/deserialize object configs via registry
"""

from __future__ import annotations

import inspect
import json
from typing import Dict, Type, Callable, Optional, Tuple, TypeVar, Any
import torch

T = TypeVar("T")


def create_registry(
    registry_name: str,
    case_insensitive: bool = False,
) -> Tuple[Dict[str, Type[T]], Callable[..., Type[T]], Callable[[str], Type[T]]]:
    """
    Create a registry system with register and get functions.

    Args:
        registry_name: Name used in error messages (e.g., "projector")
        case_insensitive: Whether to store lowercase versions of names

    Returns:
        (registry_dict, register_function, get_function)
    """

    registry: Dict[str, Type[T]] = {}

    def register(cls_or_name=None, name: Optional[str] = None):
        """Register a class in the registry. Supports multiple usage patterns.

        Usage:
            @register
            class Foo: ...

            @register("foo")
            class Foo: ...

            @register(name="foo")
            class Foo: ...
        """

        def _register(c: Type[T]) -> Type[T]:
            # Determine the name to use
            if isinstance(cls_or_name, str):
                class_name = cls_or_name
            elif name is not None:
                class_name = name
            else:
                class_name = c.__name__

            registry[class_name] = c
            if case_insensitive:
                registry[class_name.lower()] = c
            return c

        if cls_or_name is not None and not isinstance(cls_or_name, str):
            # Called as @register or register(cls)
            return _register(cls_or_name)
        else:
            # Called as @register("name") or @register(name="name")
            return _register

    def get_class(name: str) -> Type[T]:
        """Get class by name from registry."""
        if name not in registry:
            # Build readable available list without duplicates when case_insensitive
            seen = set()
            available = []
            for k in registry.keys():
                if k.lower() in seen:
                    continue
                seen.add(k.lower())
                available.append(k)
            raise ValueError(
                f"Unknown {registry_name} class: {name}. Available: {available}"
            )
        return registry[name]

    return registry, register, get_class


def capture_init_args(cls):
    """
    Decorator to capture initialization arguments of a class.

    Stores the mapping of the constructor's parameters to the values supplied
    at instantiation time into `self._init_args` for later serialization.
    """
    original_init = cls.__init__

    def new_init(self, *args, **kwargs):
        # Store all initialization arguments
        init_args: Dict[str, Any] = {}

        # Get parameter names from the original __init__ method
        sig = inspect.signature(original_init)
        param_names = list(sig.parameters.keys())[1:]  # Skip 'self'

        # Map positional args to parameter names
        for i, arg in enumerate(args):
            if i < len(param_names):
                init_args[param_names[i]] = arg

        # Add keyword args
        init_args.update(kwargs)

        self._init_args = init_args

        # Call the original __init__
        original_init(self, *args, **kwargs)

    cls.__init__ = new_init
    return cls


# -------------------------
# Serialization utilities
# -------------------------

def _encode_value(value: Any) -> Any:
    """Best-effort JSON encoding for common ML types."""
    # Primitives and None
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    # Tuples -> lists
    if isinstance(value, tuple):
        return [
            _encode_value(v) for v in value
        ]

    # Lists
    if isinstance(value, list):
        return [
            _encode_value(v) for v in value
        ]

    # Dicts
    if isinstance(value, dict):
        return {k: _encode_value(v) for k, v in value.items()}

    # torch-specific types
    if torch is not None:
        # torch.dtype
        if isinstance(value, type(getattr(torch, "float32", object))):
            # Guard: torch.dtype is not a class; rely on str(value) format
            s = str(value)
            if s.startswith("torch."):
                return {"__type__": "torch.dtype", "value": s.split(".")[-1]}

        # torch.device
        if isinstance(value, getattr(torch, "device", ())):
            return {"__type__": "torch.device", "value": str(value)}

    # Fallback to string representation
    return {"__type__": "str", "value": str(value)}


def _decode_value(value: Any) -> Any:
    """Decode values produced by _encode_value, recursively for containers."""
    # Lists: decode each element
    if isinstance(value, list):
        return [_decode_value(v) for v in value]

    # Dicts: either a typed-marker dict or a regular mapping that needs recursive decoding
    if isinstance(value, dict):
        if "__type__" in value:
            t = value.get("__type__")
            v = value.get("value")

            if t == "torch.dtype" and torch is not None:
                dtype = getattr(torch, str(v), None)
                if dtype is None:
                    raise ValueError(f"Unknown torch.dtype: {v}")
                return dtype

            if t == "torch.device" and torch is not None:
                return torch.device(v)

            if t == "str":
                return str(v)

            # Unknown type marker; return raw as-is
            return value

        # Regular dict: decode values recursively
        return {k: _decode_value(v) for k, v in value.items()}

    # Primitives and anything else: return as-is
    return value


def save_object(obj: Any, file_path: str) -> None:
    """
    Save an object's construction config to a JSON file.

    The object is expected to have been decorated with capture_init_args,
    so that `obj._init_args` exists.
    """
    class_name = obj.__class__.__name__
    init_args = getattr(obj, "_init_args", {})

    serializable_args = _encode_value(init_args)
    payload = {
        "class": class_name,
        "init_args": serializable_args,
    }

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_object(
    file_path: str,
    get_class_fn: Callable[[str], Type[T]],
    override_args: Optional[Dict[str, Any]] = None,
) -> T:
    """
    Load an object from a JSON config file previously saved by save_object.

    Args:
        file_path: Path to JSON file
        get_class_fn: Function to resolve class names from registry
        override_args: Optional dict to override stored init args

    Returns:
        Instantiated object of type T
    """
    with open(file_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    class_name = payload["class"]
    encoded_args = payload.get("init_args", {})
    init_args = _decode_value(encoded_args)

    if override_args:
        init_args.update(override_args)

    cls = get_class_fn(class_name)
    return cls(**init_args)


def dumps_object_config(obj: Any) -> str:
    """Return a JSON string with the object's class and init args."""
    class_name = obj.__class__.__name__
    init_args = getattr(obj, "_init_args", {})
    serializable_args = _encode_value(init_args)
    return json.dumps({"class": class_name, "init_args": serializable_args}, indent=2)


def loads_object_config(
    s: str,
    get_class_fn: Callable[[str], Type[T]],
    override_args: Optional[Dict[str, Any]] = None,
) -> T:
    """Instantiate an object from a JSON string produced by dumps_object_config."""
    payload = json.loads(s)
    class_name = payload["class"]
    encoded_args = payload.get("init_args", {})
    init_args = _decode_value(encoded_args)
    if override_args:
        init_args.update(override_args)
    cls = get_class_fn(class_name)
    return cls(**init_args)


# Model Registry System (case-insensitive for backward compatibility)
PROJECTOR_REGISTRY, register_model, get_projector_class = create_registry(
    "projector", case_insensitive=True
)