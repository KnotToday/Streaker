"""
make_star_catalog.py — generate star_catalog.json from Yale Bright Star Catalog via VizieR.
Run once with internet access; output is bundled with the exe.
"""
import json
import math
import re
import urllib.request
from pathlib import Path

MAG_LIMIT = 5.5   # include stars brighter than this

# Greek letter abbreviation → full name used as prefix
GREEK = {
    'Alp': 'Alpha', 'Bet': 'Beta',  'Gam': 'Gamma', 'Del': 'Delta',
    'Eps': 'Eps',   'Zet': 'Zeta',  'Eta': 'Eta',   'The': 'Theta',
    'Iot': 'Iota',  'Kap': 'Kappa', 'Lam': 'Lam',   'Mu':  'Mu',
    'Nu':  'Nu',    'Xi':  'Xi',    'Omi': 'Omi',   'Pi':  'Pi',
    'Rho': 'Rho',   'Sig': 'Sigma', 'Tau': 'Tau',   'Ups': 'Ups',
    'Phi': 'Phi',   'Chi': 'Chi',   'Psi': 'Psi',   'Ome': 'Omega',
}

# HR number → IAU proper name (authoritative override)
# Sources: IAU Working Group on Star Names official list
PROPER_NAMES = {
    7001: 'Vega',          7557: 'Altair',        7924: 'Deneb',
    5340: 'Arcturus',      6134: 'Antares',        3982: 'Regulus',
    5056: 'Spica',         2491: 'Sirius',         2943: 'Procyon',
    2990: 'Pollux',        1708: 'Capella',        1457: 'Aldebaran',
    2061: 'Betelgeuse',    1713: 'Rigel',          4301: 'Dubhe',
    5191: 'Alkaid',        4905: 'Alioth',         4301: 'Dubhe',
    6527: 'Shaula',        7790: 'Peacock',        8728: 'Fomalhaut',
    8425: 'Alnair',        6879: 'Kaus Australis', 7001: 'Vega',
    4730: 'Acrux',         4763: 'Gacrux',         4853: 'Mimosa',
    5459: 'Rigil Kentaurus', 5460: 'Toliman',      5267: 'Hadar',
    6217: 'Atria',         6553: 'Sargas',         6134: 'Antares',
    3685: 'Miaplacidus',   3207: 'Regor',
    # Northern/circumpolar
    424:  'Mirach',        15:   'Alpheratz',       264:  'Almach',
    168:  'Almaak',
    1790: 'Bellatrix',     1903: 'Alnilam',        1948: 'Mintaka',
    2088: 'Menkalinan',    2294: 'Murzim',         2618: 'Adhara',
    2693: 'Wezen',         2421: 'Alhena',         2891: 'Castor',
    1791: 'Elnath',        1791: 'Alnath',
    3748: 'Alphard',
    472:  'Achernar',      5054: 'Mizar',          5054: 'Mizar',
    4660: 'Megrez',        4301: 'Dubhe',          4295: 'Merak',
    4905: 'Alkaid',        4905: 'Alkaid',
    # Cygnus
    7924: 'Deneb',         7796: 'Sadr',           7528: 'Fawaris',
    7949: 'Aljanah',       7417: 'Albireo',
    # Lyra
    7001: 'Vega',          7106: 'Sulafat',        7178: 'Sheliak',
    # Aquila
    7557: 'Altair',        7525: 'Tarazed',        7602: 'Alshain',
    7595: 'Deneb Okab',
    # Hercules
    6148: 'Kornephoros',   6212: 'Sarin',          6603: 'Rasalgethi',
    6418: 'Zeta Her',      6458: 'Pi Her',         6623: 'Mu Her',
    6945: 'Eta Her',
    # Boötes
    5340: 'Arcturus',      5235: 'Izar',           5602: 'Muphrid',
    5478: 'Nekkar',        5435: 'Seginus',        5404: 'Nusakan',
    # Ophiuchus
    6556: 'Rasalhague',    6299: 'Cebalrai',       6075: 'Yed Prior',
    6123: 'Yed Posterior',
    # Scorpius
    6134: 'Antares',       6527: 'Shaula',         6553: 'Sargas',
    6165: 'Graffias',      6084: 'Dschubba',
    # Sagittarius
    6913: 'Kaus Borealis', 6879: 'Kaus Australis', 6859: 'Kaus Media',
    7194: 'Nunki',         7121: 'Ascella',        7348: 'Albaldah',
    7264: 'Phi Sagittarii',
    # Scorpius continued
    6247: 'Al Niyat',      6241: 'Al Niyat',
    # Draco
    6705: 'Eltanin',       6536: 'Rastaban',       5291: 'Thuban',
    6636: 'Altais',        5744: 'Edasich',
    # Ursa Minor
    424:  'Pherkad',       5563: 'Kochab',
    # Cassiopeia
    168:  'Schedar',       21:   'Caph',            403:  'Ruchbah',
    542:  'Segin',
    # Perseus
    1017: 'Mirfak',        921:  'Algol',
    # Taurus
    1457: 'Aldebaran',     1411: 'Alcyone',        1165: 'Celaeno',
    1142: 'Electra',       1156: 'Taygeta',        1178: 'Maia',
    1279: 'Merope',        1409: 'Atlas',
    1346: 'Tianguan',      1373: 'Propus',
    # Gemini
    2891: 'Castor',        2990: 'Pollux',         2697: 'Alhena',
    2650: 'Mebsuda',       2763: 'Tejat',
    # Orion
    2061: 'Betelgeuse',    1713: 'Rigel',          1790: 'Bellatrix',
    1852: 'Mintaka',       1903: 'Alnilam',        1948: 'Alnitak',
    2004: 'Saiph',
    # Leo
    3982: 'Regulus',       4057: 'Algieba',        4357: 'Denebola',
    4031: 'Zosma',         3975: 'Adhafera',
    # Virgo
    5056: 'Spica',         4932: 'Porrima',        5338: 'Minelauva',
    # Libra
    5531: 'Zubenelgenubi', 5603: 'Zubeneschamali',
    # Corona Borealis
    5793: 'Alphecca',      5747: 'Nusakan',
    # Serpens
    5854: 'Unukalhai',
    # Aquarius
    8414: 'Sadalsuud',     8232: 'Sadalmelik',     8709: 'Skat',
    # Pegasus
    8775: 'Scheat',        8781: 'Markab',         8634: 'Matar',
    # Andromeda
    15:   'Alpheratz',     337:  'Mirach',         603:  'Almach',
    # Auriga
    1708: 'Capella',       1641: 'Menkalinan',     1577: 'Mahasim',
    # Cepheus
    8162: 'Alderamin',     8238: 'Alfirk',         8974: 'Errai',
    # Camelopardalis/others
    1203: 'Cursa',
    # Eridanus
    1084: 'Acamar',        897:  'Zaurak',
    # Canis Major
    2491: 'Sirius',        2618: 'Adhara',         2693: 'Wezen',
    2294: 'Murzim',
    # Canis Minor
    2943: 'Procyon',       2845: 'Gomeisa',
    # Aries
    617:  'Hamal',         553:  'Sheratan',
    # Pisces
    596:  'Alrescha',
    # Cetus
    911:  'Diphda',        681:  'Menkar',
    # Columba
    1956: 'Phact',
    # Centaurus
    5267: 'Hadar',         5459: 'Rigil Kentaurus',
}


def parse_ra(hms):
    """Parse 'HH MM SS.S' to decimal degrees."""
    hms = hms.strip()
    parts = re.split(r'[\s:]+', hms)
    h, m, s = float(parts[0]), float(parts[1]), float(parts[2])
    return (h + m / 60 + s / 3600) * 15.0


def parse_dec(dms):
    """Parse '+DD MM SS' to decimal degrees."""
    dms = dms.strip()
    sign = -1 if dms.startswith('-') else 1
    dms = dms.lstrip('+-')
    parts = re.split(r'[\s:]+', dms)
    d, m, s = float(parts[0]), float(parts[1]), float(parts[2])
    return sign * (d + m / 60 + s / 3600)


def bayer_to_name(raw, con):
    """Convert 'Alp', 'Bet1', etc. + constellation to a display name."""
    raw = raw.strip()
    # Extract greek prefix
    m = re.match(r'^([A-Za-z]+?)(\d*)$', raw)
    if not m:
        return None
    greek_abbr = m.group(1)
    number = m.group(2)
    if greek_abbr not in GREEK:
        return None
    greek = GREEK[greek_abbr]
    suffix = number if number else ''
    return f'{greek}{suffix} {con.strip()}'


def fetch_catalog(mag_limit=5.5):
    url = (f'https://vizier.cds.unistra.fr/viz-bin/asu-tsv'
           f'?-source=V/50&-out=HR,Name,RAJ2000,DEJ2000,Vmag'
           f'&Vmag=<{mag_limit}&-out.max=99999')
    print(f'Fetching Yale BSC from VizieR (mag < {mag_limit})...')
    with urllib.request.urlopen(url, timeout=60) as r:
        raw = r.read().decode('latin-1')

    stars = []
    seen_hr = set()

    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('-'):
            continue
        parts = line.split('\t')
        if len(parts) < 5:
            continue
        hr_str, name_str, ra_str, dec_str, mag_str = parts[:5]

        # Skip header lines
        try:
            hr = int(hr_str.strip())
        except ValueError:
            continue
        if hr in seen_hr:
            continue
        seen_hr.add(hr)

        try:
            mag = float(mag_str.strip())
        except ValueError:
            continue
        if mag > mag_limit:
            continue

        try:
            ra = parse_ra(ra_str)
            dec = parse_dec(dec_str)
        except (ValueError, IndexError):
            continue

        # Extract Flamsteed number from name field if present
        flamsteed = None
        nm_raw = name_str.strip()
        fl_m = re.match(r'^\s*(\d+)\s*', nm_raw)
        if fl_m:
            flamsteed = int(fl_m.group(1))

        # Determine display name
        if hr in PROPER_NAMES:
            name = PROPER_NAMES[hr]
        else:
            # Parse Bayer from name field: "33Alp Per" or " 9Alp CMa" or "Alp Eri"
            nm = re.sub(r'^\s*\d+\s*', '', nm_raw)
            m2 = re.match(r'^([A-Za-z]+\d*)\s+([A-Za-z]+\d*)$', nm)
            if m2:
                name = bayer_to_name(m2.group(1), m2.group(2))
            else:
                name = None

        if not name:
            continue  # skip Flamsteed-only stars without Bayer designation

        entry = {'name': name, 'ra': round(ra, 4),
                 'dec': round(dec, 4), 'mag': round(mag, 2), 'hr': hr}
        if flamsteed:
            entry['fl'] = flamsteed
        stars.append(entry)

    # Sort by magnitude
    stars.sort(key=lambda s: s['mag'])
    print(f'Generated {len(stars)} catalog entries.')
    return stars


if __name__ == '__main__':
    stars = fetch_catalog(MAG_LIMIT)
    out = Path(__file__).parent / 'star_catalog.json'
    out.write_text(json.dumps(stars, indent=1))
    print(f'Saved to {out}')
    # Show sample
    for s in stars[:10]:
        print(f"  {s['name']:25s}  RA {s['ra']:8.4f}  Dec {s['dec']:+8.4f}  mag {s['mag']}")
