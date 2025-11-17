import nanoid

alpha36 = "0123456789abcdefghijklmnopqrstuvwxyz"
alpha16 = "0123456789abcdef"
types = "dfce"  # document, file, chunk, entity

docid_nanoid_chars = 23  # 36^23 is a bit less than 2^119 (~15 bytes)

def nanoid36() -> str:
    """
    Free of punctuation and uppercase; still as good as UUID4.
    """
    return nanoid.generate(alpha36, docid_nanoid_chars)

def mkdocid(code: str = "d") -> str:
    """
    Docid that qualifies as a URI with aryn: scheme.
    """
    return f"aryn:{code}-{nanoid36()}"