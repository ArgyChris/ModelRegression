from pydantic import BaseModel, Field

# Define Pydantic model for the input data with available data from the California Housing Market
class HousingFeatures(BaseModel):
    longitude: float = Field(-118.49, ge=-124.35, le=-114.31)  # Restrict in the bounding box of California state
    latitude: float = Field(34.26, ge=32.54, le=41.95)  # Restrict in the bounding box of California state
    housing_median_age: float = Field(29.0, ge=1.0, le=52)
    total_rooms: float = Field(2127.0, ge=2.0, le=39320.0)
    total_bedrooms: float = Field(537, ge=1.0, le=6445.0)
    population: float = Field(1425, ge=3.0, le=35682.0) # Population should be non-negative
    households: float = Field(409, ge=1.0, le=6082.0) # Households should be positive
    median_income: float = Field(3.5, ge=0.4999, le=15.000) # Median income should be positive
    ocean_proximity: float= Field(1, ge=0, le=5) # Ocean proximity can be any value between 0 and 5
    
    