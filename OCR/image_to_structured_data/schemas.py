from pydantic import BaseModel, Field
from typing import List, Optional

class ProductAttribute(BaseModel):
    key: str = Field(description="The attribute name, e.g., Color, Material, Size")
    value: str

class StructuredProduct(BaseModel):
    """Extraction schema for product labels or advertisements."""
    name: str = Field(description="The primary name of the product")
    brand: Optional[str]
    price: Optional[float]
    currency: Optional[str] = Field(description="ISO currency code")
    attributes: List[ProductAttribute]
    summary: str = Field(description="A brief 1-sentence description of the visual style")

class InvoiceData(BaseModel):
    """Extraction schema for invoices or receipts."""
    vendor_name: str
    date: str
    total_amount: float
    items: List[str]