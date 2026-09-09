import requests
import xml.etree.ElementTree as ET
import urllib3

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Test ArXiv API call
api_url = "https://export.arxiv.org/api/query?id_list=2301.07041"
response = requests.get(api_url, verify=False, timeout=10)

print("=== ArXiv API Response Analysis ===")
print(f"Status Code: {response.status_code}")
print(f"Content Type: {response.headers.get('content-type', 'N/A')}")
print("\n=== XML Content ===")
print(response.text)

# Parse and analyze XML structure
try:
    root = ET.fromstring(response.content)
    
    print("\n\n=== Available XML Elements ===")
    ns = {
        'atom': 'http://www.w3.org/2005/Atom',
        'arxiv': 'http://arxiv.org/schemas/atom'
    }
    
    entry = root.find('.//atom:entry', ns)
    if entry is not None:
        print("Found entry element")
        for child in entry:
            print(f"- {child.tag}: {child.text[:100] if child.text else 'N/A'}...")
            # Print attributes if any
            if child.attrib:
                print(f"  Attributes: {child.attrib}")
            
            # Check for nested elements
            if len(child) > 0:
                print(f"  Nested elements:")
                for nested in child:
                    print(f"    - {nested.tag}: {nested.text[:50] if nested.text else 'N/A'}...")
                    if nested.attrib:
                        print(f"      Attributes: {nested.attrib}")

except Exception as e:
    print(f"Error parsing XML: {e}")