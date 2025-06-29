from geopy.geocoders import Nominatim
from geopy.distance import geodesic

def get_coordinates_from_place(place_name):
    """
    Geocode a place name to (latitude, longitude) using Nominatim.
    Returns (lat, lon) tuple if found, else None.
    """
    try:
        geolocator = Nominatim(user_agent="EchoWear_Navigation_App", timeout=10)
        location = geolocator.geocode(place_name)
        if location:
            return (location.latitude, location.longitude)
        else:
            return None
    except Exception as e:
        print(f"Geocoding error: {e}")
        return None

def reverse_geocode(latitude, longitude):
    """
    Reverse geocode coordinates to find a human-readable address or place name.
    Returns the address string or None if not found.
    """
    try:
        geolocator = Nominatim(user_agent="EchoWear_Navigation_App", timeout=10)
        location = geolocator.reverse((latitude, longitude))
        if location:
            return location.address
        else:
            return None
    except Exception as e:
        print(f"Reverse geocoding error: {e}")
        return None

def find_nearby_places(latitude, longitude, query=None, distance_km=0.5):
    """
    Find places near the given coordinates using Nominatim.
    
    Args:
        latitude (float): Latitude coordinate
        longitude (float): Longitude coordinate
        query (str, optional): Specific type of place to search for (e.g., "cafe", "park")
        distance_km (float): Search radius in kilometers (default: 0.5)
    
    Returns:
        list: List of dictionaries containing 'name' and 'address' of found places
    """
    try:
        geolocator = Nominatim(user_agent="EchoWear_Navigation_App", timeout=15)
        
        # Create search query
        if query:
            search_query = f"{query} near {latitude}, {longitude}"
        else:
            search_query = f"near {latitude}, {longitude}"
        
        # Search for places
        locations = geolocator.geocode(search_query, exactly_one=False, limit=10)
        
        nearby_places = []
        if locations:
            for location in locations:
                # Calculate distance from the given coordinates
                distance = geodesic((latitude, longitude), (location.latitude, location.longitude)).kilometers
                
                # Only include places within the specified distance
                if distance <= distance_km:
                    nearby_places.append({
                        'name': location.raw.get('display_name', '').split(',')[0],
                        'address': location.address,
                        'distance_km': round(distance, 3),
                        'coordinates': (location.latitude, location.longitude)
                    })
        
        # Sort by distance
        nearby_places.sort(key=lambda x: x['distance_km'])
        return nearby_places
        
    except Exception as e:
        print(f"Error finding nearby places: {e}")
        return []

if __name__ == "__main__":
    print("=== Geocoding Utils Demo ===\n")
    
    # Example 1: Reverse geocoding (Eiffel Tower coordinates)
    print("1. Reverse Geocoding Example:")
    eiffel_coords = (48.8584, 2.2945)  # Eiffel Tower coordinates
    address = reverse_geocode(*eiffel_coords)
    if address:
        print(f"   Coordinates {eiffel_coords} -> {address}")
    else:
        print(f"   Could not find address for coordinates {eiffel_coords}")
    
    print()
    
    # Example 2: Finding nearby restaurants/cafes
    print("2. Finding Nearby Places Example:")
    # Using Times Square coordinates as an example
    times_square_coords = (40.7580, -73.9855)
    nearby_places = find_nearby_places(*times_square_coords, query="restaurant", distance_km=1.0)
    
    if nearby_places:
        print(f"   Found {len(nearby_places)} restaurants near Times Square:")
        for i, place in enumerate(nearby_places[:5], 1):  # Show first 5 results
            print(f"   {i}. {place['name']} ({place['distance_km']} km)")
            print(f"      Address: {place['address']}")
    else:
        print("   No nearby restaurants found")
    
    print()
    
    # Example 3: Interactive geocoding (original functionality)
    print("3. Interactive Geocoding:")
    place = input("Enter a place name to geocode: ")
    coords = get_coordinates_from_place(place)
    if coords:
        print(f"   Coordinates for '{place}': {coords}")
    else:
        print(f"   Could not find coordinates for '{place}'") 