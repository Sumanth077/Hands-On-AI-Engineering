from pydantic import BaseModel, Field
from typing import List, Optional

class ProductAttribute(BaseModel):
    """A single key-value attribute describing a product."""
    key: str
    value: str

class StructuredProduct(BaseModel):
    """A single product extracted from an image, with pricing and attributes."""
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
    """A single invoice extracted from an image, with vendor, total, and line items."""
    vendor_name: str
    date: str
    total_amount: float
    items: List[str]

class InvoiceCollection(BaseModel):
    """A collection of invoices or line items found."""
    invoices: List[InvoiceData]