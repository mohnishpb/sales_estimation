# config.py

"""
Configuration file for the vehicle price prediction model.

This file contains a static map (`BRAND_TIER_MAP`) for classifying vehicle makes
into tiers based on a combination of domain knowledge and data analysis.

The tiers are defined as follows:
- Exotic: Ultra-luxury, low-volume supercars.
- Luxury: High-end, established luxury automotive brands.
- Premium: Brands positioned above mass-market but below true luxury.
- Standard: Mass-market, high-volume brands.

NOTE: The keys in this dictionary should match the 'Lot Make' values AFTER they have
been cleaned (i.e., whitespace has been stripped).
"""

BRAND_TIER_MAP = {
    # ==========================================================================
    # Tier 1: Exotic Brands
    # Justification: Highest mean/median sale prices (> $45k) in the provided data.
    # These are low-volume, ultra-high-performance vehicles.
    # ==========================================================================
    'LAMO': 'Exotic',    # Lamborghini
    'ROL':  'Exotic',    # Rolls-Royce
    'FERR': 'Exotic',    # Ferrari
    'MCLA': 'Exotic',    # McLaren

    # ==========================================================================
    # Tier 2: Luxury Brands
    # Justification: High mean prices, high max values, and strong brand recognition.
    # These brands often have a very wide price range.
    # ==========================================================================
    'BENT': 'Luxury',    # Bentley (Mean ~$44k)
    'ASTO': 'Luxury',    # Aston Martin (Mean ~$19k but strong luxury brand)
    'PORS': 'Luxury',    # Porsche (Mean ~$18k but max >$120k, classic luxury)
    'MASE': 'Luxury',    # Maserati
    'MERZ': 'Luxury',    # Mercedes-Benz (Mean is low due to volume, but max is high)
    'JAGR': 'Luxury',    # Jaguar (Spelled JAGU in your list)

    # ==========================================================================
    # Tier 3: Premium Brands
    # Justification: Positioned above standard brands. Includes entry-level luxury
    # and brands with higher average sale prices than mass-market cars.
    # ==========================================================================
    'TESL': 'Premium',   # Tesla (Mean ~$10k)
    'LAND': 'Premium',   # Land Rover (Mean ~$10k)
    'GENS': 'Premium',   # Genesis (Mean ~$8.5k)
    'AUDI': 'Premium',
    'BMW':  'Premium',
    'LEXS': 'Premium',
    'ACUR': 'Premium',
    'INFI': 'Premium',
    'VOLV': 'Premium',
    'ALFA': 'Premium',   # Alfa Romeo
    'CADI': 'Premium',   # Cadillac
    'LINC': 'Premium',   # Lincoln
    'FISK': 'Premium',   # Fisker (Mean ~$11k)
    'KARM': 'Premium',   # Karma (Mean ~$17k)

    # ==========================================================================
    # Tier 4: Standard Brands
    # Justification: High-volume, mass-market brands with lower average prices.
    # We only list the most common ones; the dynamic system will handle the rest.
    # ==========================================================================
    'TOYT': 'Standard',  # Toyota
    'HOND': 'Standard',  # Honda
    'FORD': 'Standard',
    'CHEV': 'Standard',  # Chevrolet
    'NISS': 'Standard',  # Nissan
    'HYUN': 'Standard',  # Hyundai
    'KIA':  'Standard',
    'SUBA': 'Standard',  # Subaru
    'MAZD': 'Standard',  # Mazda
    'VOLK': 'Standard',  # Volkswagen
    'JEP':  'Standard',  # Jeep
    'GMC':  'Standard',
    'DODG': 'Standard',
    'RAM':  'Standard',
    'CHRY': 'Standard',  # Chrysler
    'MITS': 'Standard',  # Mitsubishi
    'BUIC': 'Standard',  # Buick
    'FIAT': 'Standard',
    'SCIO': 'Standard',  # Scion
    'SATU': 'Standard',  # Saturn
}

# ==============================================================================
# NOTE ON UNCATEGORIZED BRANDS
#
# The following types of brands are deliberately LEFT OUT of this static map.
# The `train.py` script will use its dynamic, price-based heuristic to assign
# them a tier during each retraining run. This makes the system more robust.
#
# - Commercial Trucks: KW (Kenworth), PTRB (Peterbilt), FRHT (Freightliner), INTL
# - Motorcycles/Powersports: HARL, YAMA, KAWA, SUZI, CANA, POLS, TRIU
# - Extremely Rare/Ambiguous: SAND, SPCN, ROAR, LANC, etc. (brands with very
#   few data points are better handled dynamically).
# - "Other" Categories: OTHE, OTHR
# ==============================================================================