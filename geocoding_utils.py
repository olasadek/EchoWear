from geopy.geocoders import Nominatim

def get_coordinates_from_place(place_name):
    """
    Geocode a place name to (latitude, longitude) using Nominatim.
    Returns (lat, lon) tuple if found, else None.
    """
    try:
        geolocator = Nominatim(user_agent="EchoWear_Navigation_App")
        location = geolocator.geocode(place_name)
        if location:
            return (location.latitude, location.longitude)
        else:
            return None
    except Exception as e:
        print(f"Geocoding error: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    place = input("Enter a place name to geocode: ")
    coords = get_coordinates_from_place(place)
    if coords:
        print(f"Coordinates for '{place}': {coords}")
    else:
        print(f"Could not find coordinates for '{place}'") 