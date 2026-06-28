from pydantic import BaseModel, Field
from typing import List, Optional

class ProductAttribute(BaseModel):
<<<<<<< HEAD
=======
    """A single key-value attribute describing a product."""
>>>>>>> 1d1e9f137cfd1123edbae5d8e955ce0b9c7fcf4a
    key: str
    value: str

class StructuredProduct(BaseModel):
<<<<<<< HEAD
=======
    """A single product extracted from an image, with pricing and attributes."""
>>>>>>> 1d1e9f137cfd1123edbae5d8e955ce0b9c7fcf4a
    name: str
    brand: Optional[str]
    price: Optional[float]
    currency: Optional[str]
    attributes: List[ProductAttribute]
    summary: str

# These are the ones the error is looking for:
class ProductCollection(BaseModel):
    """A collection of all products found in the image."""
    products: List[StructuredProduct]

class InvoiceData(BaseModel):
<<<<<<< HEAD
=======
    """A single invoice extracted from an image, with vendor, total, and line items."""
>>>>>>> 1d1e9f137cfd1123edbae5d8e955ce0b9c7fcf4a
    vendor_name: str
    date: str
    total_amount: float
    items: List[str]

class InvoiceCollection(BaseModel):
    """A collection of invoices or line items found."""
    invoices: List[InvoiceData]