import time
import uuid

def ts_suffix(prefix: str = "") -> str:
    return f"{prefix}{int(time.time())}-{uuid.uuid4().hex[:8]}"

def sanitize(name: str) -> str:
    x = "".join(c if c.isalnum() or c == "-" else "-" for c in name.lower())
    return x[:63].rstrip("-")
