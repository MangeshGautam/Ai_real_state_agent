from fastapi import APIRouter
from app.Models.request import PropertySearchRequest
from app.Services.real_estate import run_sequential_analysis
from app.core.config import GOOGLE_API_KEY, FIRECRAWL_API_KEY

router = APIRouter()

@router.post("/properties/search")
def search_properties(payload: PropertySearchRequest):

    user_criteria = {
        "budget_range": f"${payload.min_price:,} - ${payload.max_price:,}",
        "property_type": payload.property_type,
        "bedrooms": payload.bedrooms,
        "bathrooms": payload.bathrooms,
        "min_sqft": payload.min_sqft,
        "special_features": payload.special_features
    }

    result = run_sequential_analysis(
        city=payload.city,
        state=payload.state,
        user_criteria=user_criteria,
        selected_websites=payload.selected_websites,
        firecrawl_api_key=FIRECRAWL_API_KEY,
        google_api_key=GOOGLE_API_KEY,
        update_callback=lambda *args: None  # 🚨 REMOVE UI CALLBACK
    )

    return result
