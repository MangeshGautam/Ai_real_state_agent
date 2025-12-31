from pydantic import BaseModel
from typing import List

class PropertySearchRequest(BaseModel):
    city: str
    state: str | None
    min_price: int
    max_price: int
    property_type: str
    bedrooms: str
    bathrooms: str
    min_sqft: int
    special_features: str
    selected_websites: List[str]
