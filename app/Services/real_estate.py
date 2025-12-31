import json
import time
import re
from agno.agent import Agent
from agno.run.agent import RunOutput
from agno.models.google import Gemini
from dotenv import load_dotenv
from firecrawl import FirecrawlApp
from pydantic import BaseModel, Field
from typing import List, Optional
from app.core.config import GOOGLE_API_KEY, FIRECRAWL_API_KEY



class PropertyDetails(BaseModel):
    address: Optional[str]
    price: Optional[str] = "Not specified"
    bedrooms: Optional[str] = "Not specified"
    bathrooms: Optional[str] = "Not specified"
    square_feet: Optional[str] = "Not specified"
    property_type: Optional[str] = "Not specified"
    description: Optional[str] = "Not specified"
    features: Optional[List[str]]
    images: Optional[List[str]]
    listing_url: Optional[str] = None
    agent_contact: Optional[str] = "Not available"


class PropertyListing(BaseModel):
    properties: List[PropertyDetails]
    total_count: int
    source_website: Optional[str]


class DirectFirecrawlAgent:
    """Agent with direct Firecrawl integration for property search"""
    
    def __init__(self, firecrawl_api_key: str, google_api_key: str, model_id: str = "gemini-2.5-flash"):
        self.agent = Agent(
            model=Gemini(id=model_id, api_key=google_api_key),
            markdown=True,
            description="I am a real estate expert who helps find and analyze properties based on user preferences."
        )
        self.firecrawl = FirecrawlApp(api_key=firecrawl_api_key)

    def find_properties_direct(self, city: str, state: str, user_criteria: dict, selected_websites: list) -> dict:
        """Direct Firecrawl integration for property search"""
        city_formatted = city.replace(' ', '-').lower()
        state_upper = state.upper() if state else ''
        
        # Create URLs for selected websites
        state_lower = state.lower() if state else ''
        city_trulia = city.replace(' ', '_')  # Trulia uses underscores for spaces
        search_urls = {
            "Zillow": f"https://www.zillow.com/homes/for_sale/{city_formatted}-{state_upper}/",
            "Realtor.com": f"https://www.realtor.com/realestateandhomes-search/{city_formatted}_{state_upper}/pg-1",
            "Trulia": f"https://www.trulia.com/{state_upper}/{city_trulia}/",
            "Homes.com": f"https://www.homes.com/homes-for-sale/{city_formatted}-{state_lower}/"
        }
        
        # Filter URLs based on selected websites
        urls_to_search = [url for site, url in search_urls.items() if site in selected_websites]
        
        print(f"Selected websites: {selected_websites}")
        print(f"URLs to search: {urls_to_search}")
        
        if not urls_to_search:
            return {"error": "No websites selected"}
        
        # Create comprehensive prompt with specific schema guidance
        prompt = f"""You are extracting property listings from real estate websites. Extract EVERY property listing you can find on the page.

USER SEARCH CRITERIA:
- Budget: {user_criteria.get('budget_range', 'Any')}
- Property Type: {user_criteria.get('property_type', 'Any')}
- Bedrooms: {user_criteria.get('bedrooms', 'Any')}
- Bathrooms: {user_criteria.get('bathrooms', 'Any')}
- Min Square Feet: {user_criteria.get('min_sqft', 'Any')}
- Special Features: {user_criteria.get('special_features', 'Any')}

EXTRACTION INSTRUCTIONS:
1. Find ALL property listings on the page (usually 20-40 per page)
2. For EACH property, extract these fields:
   - address: Full street address (required)
   - price: Listed price with $ symbol (required) 
   - bedrooms: Number of bedrooms (required)
   - bathrooms: Number of bathrooms (required)
   - square_feet: Square footage if available
   - property_type: House/Condo/Townhouse/Apartment etc.
   - description: Brief property description if available
   - listing_url: Direct link to property details if available
   - agent_contact: Agent name/phone if visible

3. CRITICAL REQUIREMENTS:
   - Extract AT LEAST 10 properties if they exist on the page
   - Do NOT skip properties even if some fields are missing
   - Use "Not specified" for missing optional fields
   - Ensure address and price are always filled
   - Look for property cards, listings, search results

4. RETURN FORMAT:
   - Return JSON with "properties" array containing all extracted properties
   - Each property should be a complete object with all available fields
   - Set "total_count" to the number of properties extracted
   - Set "source_website" to the main website name (Zillow/Realtor/Trulia/Homes)

EXTRACT EVERY VISIBLE PROPERTY LISTING - DO NOT LIMIT TO JUST A FEW!
        """
        
        try:
            # Direct Firecrawl call - using correct API format
            print(f"Calling Firecrawl with {len(urls_to_search)} URLs")
            raw_response = self.firecrawl.extract(
                urls_to_search,
                prompt=prompt,
                schema=PropertyListing.model_json_schema()
            )
            
            print("Raw Firecrawl Response:", raw_response)
            
            if hasattr(raw_response, 'success') and raw_response.success:
                # Handle Firecrawl response object
                properties = raw_response.data.get('properties', []) if hasattr(raw_response, 'data') else []
                total_count = raw_response.data.get('total_count', 0) if hasattr(raw_response, 'data') else 0
                print(f"Response data keys: {list(raw_response.data.keys()) if hasattr(raw_response, 'data') else 'No data'}")
            elif isinstance(raw_response, dict) and raw_response.get('success'):
                # Handle dictionary response
                properties = raw_response['data'].get('properties', [])
                total_count = raw_response['data'].get('total_count', 0)
                print(f"Response data keys: {list(raw_response['data'].keys())}")
            else:
                properties = []
                total_count = 0
                print(f"Response failed or unexpected format: {type(raw_response)}")
            
            print(f"Extracted {len(properties)} properties from {total_count} total found")
            
            # Debug: Print first property if available
            if properties:
                print(f"First property sample: {properties[0]}")
                return {
                    'success': True,
                    'properties': properties,
                    'total_count': len(properties),
                    'source_websites': selected_websites
                }
            else:
                # Enhanced error message with debugging info
                error_msg = f"""No properties extracted despite finding {total_count} listings.
                
                POSSIBLE CAUSES:
                1. Website structure changed - extraction schema doesn't match
                2. Website blocking or requiring interaction (captcha, login)
                3. Properties don't match specified criteria too strictly
                4. Extraction prompt needs refinement for this website
                
                SUGGESTIONS:
                - Try different websites (Zillow, Realtor.com, Trulia, Homes.com)
                - Broaden search criteria (Any bedrooms, Any type, etc.)
                - Check if website requires specific user interaction
                
                Debug Info: Found {total_count} listings but extraction returned empty array."""
                
                return {"error": error_msg}
                
        except Exception as e:
            return {"error": f"Firecrawl extraction failed: {str(e)}"}




def create_sequential_agents(llm, user_criteria):
    """Create agents for sequential manual execution"""
    
    property_search_agent = Agent(
        name="Property Search Agent",
        model=llm,
        instructions="""
        You are a property search expert. Your role is to find and extract property listings.
        
        WORKFLOW:
        1. SEARCH FOR PROPERTIES:
           - Use the provided Firecrawl data to extract property listings
           - Focus on properties matching user criteria
           - Extract detailed property information
        
        2. EXTRACT PROPERTY DATA:
           - Address, price, bedrooms, bathrooms, square footage
           - Property type, features, listing URLs
           - Agent contact information
        
        3. PROVIDE STRUCTURED OUTPUT:
           - List properties with complete details
           - Include all listing URLs
           - Rank by match quality to user criteria
        
        IMPORTANT: 
        - Focus ONLY on finding and extracting property data
        - Do NOT provide market analysis or valuations
        - Your output will be used by other agents for analysis
        """,
    )
    
    market_analysis_agent = Agent(
        name="Market Analysis Agent",
        model=llm,
        instructions="""
        You are a market analysis expert. Provide CONCISE market insights.
        
        REQUIREMENTS:
        - Keep analysis brief and to the point
        - Focus on key market trends only
        - Provide 2-3 bullet points per area
        - Avoid repetition and lengthy explanations
        
        COVER:
        1. Market Condition: Buyer's/seller's market, price trends
        2. Key Neighborhoods: Brief overview of areas where properties are located
        3. Investment Outlook: 2-3 key points about investment potential
        
        FORMAT: Use bullet points and keep each section under 100 words.
        """,
    )
    
    property_valuation_agent = Agent(
        name="Property Valuation Agent",
        model=llm,
        instructions="""
        You are a property valuation expert. Provide CONCISE property assessments.
        
        REQUIREMENTS:
        - Keep each property assessment brief (2-3 sentences max)
        - Focus on key points only: value, investment potential, recommendation
        - Avoid lengthy analysis and repetition
        - Use bullet points for clarity
        
        FOR EACH PROPERTY, PROVIDE:
        1. Value Assessment: Fair price, over/under priced
        2. Investment Potential: High/Medium/Low with brief reason
        3. Key Recommendation: One actionable insight
        
        FORMAT: 
        - Use bullet points
        - Keep each property under 50 words
        - Focus on actionable insights only
        """,
    )
    
    return property_search_agent, market_analysis_agent, property_valuation_agent

def run_sequential_analysis(city, state, user_criteria, selected_websites, firecrawl_api_key, google_api_key, update_callback):
    """Run agents sequentially with manual coordination"""
    
    # Initialize agents
    llm = Gemini(id="gemini-2.5-flash", api_key=google_api_key)
    property_search_agent, market_analysis_agent, property_valuation_agent = create_sequential_agents(llm, user_criteria)
    
    # Step 1: Property Search with Direct Firecrawl Integration
    update_callback(0.2, "Searching properties...", "🔍 Property Search Agent: Finding properties...")
    
    direct_agent = DirectFirecrawlAgent(
        firecrawl_api_key=firecrawl_api_key,
        google_api_key=google_api_key,
        model_id="gemini-2.5-flash"
    )

    print("direct_agent", direct_agent)
    
    properties_data = direct_agent.find_properties_direct(
        city=city,
        state=state,
        user_criteria=user_criteria,
        selected_websites=selected_websites
    )
    
    if "error" in properties_data:
        return f"Error in property search: {properties_data['error']}"
    
    properties = properties_data.get('properties', [])
    if not properties:
        return "No properties found matching your criteria."
    
    update_callback(0.4, "Properties found", f"✅ Found {len(properties)} properties")
    
    # Step 2: Market Analysis
    update_callback(0.5, "Analyzing market...", "📊 Market Analysis Agent: Analyzing market trends...")
    
    market_analysis_prompt = f"""
    Provide CONCISE market analysis for these properties:
    
    PROPERTIES: {len(properties)} properties in {city}, {state}
    BUDGET: {user_criteria.get('budget_range', 'Any')}
    
    Give BRIEF insights on:
    • Market condition (buyer's/seller's market)
    • Key neighborhoods where properties are located
    • Investment outlook (2-3 bullet points max)
    
    Keep each section under 100 words. Use bullet points.
    """
    
    market_result: RunOutput = market_analysis_agent.run(market_analysis_prompt)
    market_analysis = market_result.content
    
    update_callback(0.7, "Market analysis complete", "✅ Market analysis completed")
    
    # Step 3: Property Valuation
    update_callback(0.8, "Evaluating properties...", "💰 Property Valuation Agent: Evaluating properties...")
    
    # Create detailed property list for valuation
    properties_for_valuation = []
    for i, prop in enumerate(properties, 1):
        if isinstance(prop, dict):
            prop_data = {
                'number': i,
                'address': prop.get('address', 'Address not available'),
                'price': prop.get('price', 'Price not available'),
                'property_type': prop.get('property_type', 'Type not available'),
                'bedrooms': prop.get('bedrooms', 'Not specified'),
                'bathrooms': prop.get('bathrooms', 'Not specified'),
                'square_feet': prop.get('square_feet', 'Not specified')
            }
        else:
            prop_data = {
                'number': i,
                'address': getattr(prop, 'address', 'Address not available'),
                'price': getattr(prop, 'price', 'Price not available'),
                'property_type': getattr(prop, 'property_type', 'Type not available'),
                'bedrooms': getattr(prop, 'bedrooms', 'Not specified'),
                'bathrooms': getattr(prop, 'bathrooms', 'Not specified'),
                'square_feet': getattr(prop, 'square_feet', 'Not specified')
            }
        properties_for_valuation.append(prop_data)
    
    valuation_prompt = f"""
    Provide CONCISE property assessments for each property. Use the EXACT format shown below:
    
    USER BUDGET: {user_criteria.get('budget_range', 'Any')}
    
    PROPERTIES TO EVALUATE:
    {json.dumps(properties_for_valuation, indent=2)}
    
    For EACH property, provide assessment in this EXACT format:
    
    **Property [NUMBER]: [ADDRESS]**
    • Value: [Fair price/Over priced/Under priced] - [brief reason]
    • Investment Potential: [High/Medium/Low] - [brief reason]
    • Recommendation: [One actionable insight]
    
    REQUIREMENTS:
    - Start each assessment with "**Property [NUMBER]:**"
    - Keep each property assessment under 50 words
    - Analyze ALL {len(properties)} properties individually
    - Use bullet points as shown
    """
    
    valuation_result: RunOutput = property_valuation_agent.run(valuation_prompt)
    property_valuations = valuation_result.content
    
    update_callback(0.9, "Valuation complete", "✅ Property valuations completed")
    
    # Step 4: Final Synthesis
    update_callback(0.95, "Synthesizing results...", "🤖 Synthesizing final recommendations...")
    
    # Debug: Check properties structure
    print(f"Properties type: {type(properties)}")
    print(f"Properties length: {len(properties)}")
    if properties:
        print(f"First property type: {type(properties[0])}")
        print(f"First property: {properties[0]}")
    
    # Format properties for better display
    properties_display = ""
    for i, prop in enumerate(properties, 1):
        # Handle both dict and object access
        if isinstance(prop, dict):
            address = prop.get('address', 'Address not available')
            price = prop.get('price', 'Price not available')
            prop_type = prop.get('property_type', 'Type not available')
            bedrooms = prop.get('bedrooms', 'Not specified')
            bathrooms = prop.get('bathrooms', 'Not specified')
            square_feet = prop.get('square_feet', 'Not specified')
            agent_contact = prop.get('agent_contact', 'Contact not available')
            description = prop.get('description', 'No description available')
            listing_url = prop.get('listing_url', '#')
        else:
            # Handle object access
            address = getattr(prop, 'address', 'Address not available')
            price = getattr(prop, 'price', 'Price not available')
            prop_type = getattr(prop, 'property_type', 'Type not available')
            bedrooms = getattr(prop, 'bedrooms', 'Not specified')
            bathrooms = getattr(prop, 'bathrooms', 'Not specified')
            square_feet = getattr(prop, 'square_feet', 'Not specified')
            agent_contact = getattr(prop, 'agent_contact', 'Contact not available')
            description = getattr(prop, 'description', 'No description available')
            listing_url = getattr(prop, 'listing_url', '#')
        
        properties_display += f"""




### Property {i}: {address}

**Price:** {price}  
**Type:** {prop_type}  
**Bedrooms:** {bedrooms} | **Bathrooms:** {bathrooms}  
**Square Feet:** {square_feet}  
**Agent Contact:** {agent_contact}  

**Description:** {description}  

**Listing URL:** [View Property]({listing_url})  

---
"""
    
    final_synthesis = f"""
# 🏠 Property Listings Found

**Total Properties:** {len(properties)} properties matching your criteria

{properties_display}

---

# 📊 Market Analysis & Investment Insights

        {market_analysis}

---
    
# 💰 Property Valuations & Recommendations
    
        {property_valuations}

---

# 🔗 All Property Links
    """
    
    # Extract and add property links
    all_text = f"{json.dumps(properties, indent=2)} {market_analysis} {property_valuations}"
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', all_text)
    
    if urls:
        final_synthesis += "\n### Available Property Links:\n"
        for i, url in enumerate(set(urls), 1):
            final_synthesis += f"{i}. {url}\n"
    
    update_callback(1.0, "Analysis complete", "🎉 Complete analysis ready!")
    
    # Return structured data for better UI display
    return {
        'properties': properties,
        'market_analysis': market_analysis,
        'property_valuations': property_valuations,
        'markdown_synthesis': final_synthesis,
        'total_properties': len(properties)
    }



def extract_property_valuation(property_valuations, property_number, property_address):
    """Extract valuation for a specific property from the full analysis"""
    if not property_valuations:
        return None
    
    # Split by property sections - look for the formatted property headers
    sections = property_valuations.split('**Property')
    
    # Look for the specific property number
    for section in sections:
        if section.strip().startswith(f"{property_number}:"):
            # Add back the "**Property" prefix and clean up
            clean_section = f"**Property{section}".strip()
            # Remove any extra asterisks at the end
            clean_section = clean_section.replace('**', '**').replace('***', '**')
            return clean_section
    
    # Fallback: look for property number mentions in any format
    all_sections = property_valuations.split('\n\n')
    for section in all_sections:
        if (f"Property {property_number}" in section or 
            f"#{property_number}" in section):
            return section
    
    # Last resort: try to match by address
    for section in all_sections:
        if any(word in section.lower() for word in property_address.lower().split()[:3] if len(word) > 2):
            return section
    
    # If no specific match found, return indication that analysis is not available
    return f"**Property {property_number} Analysis**\n• Analysis: Individual assessment not available\n• Recommendation: Review general market analysis in the Market Analysis tab"
