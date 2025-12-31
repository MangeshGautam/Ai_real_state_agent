from pydantic import BaseModel
from typing import List, Any

class PropertyResponse(BaseModel):
    properties: List[Any]
    market_analysis: str
    property_valuations: str
    total_properties: int
