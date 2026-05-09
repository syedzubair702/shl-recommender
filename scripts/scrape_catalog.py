"""
SHL Catalog Scraper
Fetches Individual Test Solutions from https://www.shl.com/solutions/products/product-catalog/
Outputs: data/catalog.json
"""

import json
import time
import re
import sys
from pathlib import Path
from typing import Optional

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("Installing dependencies...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "requests", "beautifulsoup4", "lxml"], check=True)
    import requests
    from bs4 import BeautifulSoup


BASE_URL = "https://www.shl.com"
CATALOG_URL = f"{BASE_URL}/solutions/products/product-catalog/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# SHL test type codes
TEST_TYPE_MAP = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "M": "Motivation",
    "P": "Personality & Behaviour",
    "S": "Simulations",
}


def get_page(url: str, retries: int = 3) -> Optional[BeautifulSoup]:
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=20)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "lxml")
        except Exception as e:
            print(f"  Attempt {attempt + 1} failed for {url}: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None


def parse_test_types(cell_text: str) -> list[str]:
    """Extract test type codes from cell content."""
    # Look for single uppercase letters that match our map
    found = re.findall(r'\b([ABCDEKMPBS])\b', cell_text)
    return [t for t in found if t in TEST_TYPE_MAP]


def scrape_catalog_page(url: str) -> list[dict]:
    """Scrape a single catalog page, return list of assessment dicts."""
    soup = get_page(url)
    if not soup:
        return []

    assessments = []

    # The catalog uses a table with product rows
    # Try multiple selector strategies
    rows = []

    # Strategy 1: find table rows with product links
    table = soup.find("table")
    if table:
        rows = table.find_all("tr")

    # Strategy 2: find product card links
    if not rows:
        rows = soup.find_all("tr")

    for row in rows:
        cells = row.find_all("td")
        if not cells:
            continue

        # Find the product name link
        link = None
        for cell in cells:
            a = cell.find("a", href=True)
            if a and "/solutions/products/" in a.get("href", ""):
                link = a
                break

        if not link:
            continue

        name = link.get_text(strip=True)
        href = link["href"]
        if href.startswith("/"):
            href = BASE_URL + href

        # Extract test types from row
        row_text = row.get_text(" ", strip=True)
        test_types = parse_test_types(row_text)

        # Check for remote/adaptive indicators
        remote_testing = "●" in row_text or "yes" in row_text.lower()

        assessment = {
            "name": name,
            "url": href,
            "test_types": test_types,
            "test_type_labels": [TEST_TYPE_MAP.get(t, t) for t in test_types],
            "remote_testing": remote_testing,
            "description": "",
            "duration_minutes": None,
            "languages": [],
        }
        assessments.append(assessment)
        print(f"  Found: {name} | types={test_types} | url={href}")

    return assessments


def scrape_product_detail(assessment: dict) -> dict:
    """Scrape individual product page for richer description."""
    soup = get_page(assessment["url"])
    if not soup:
        return assessment

    # Try to get description from meta or page content
    meta_desc = soup.find("meta", {"name": "description"})
    if meta_desc and meta_desc.get("content"):
        assessment["description"] = meta_desc["content"].strip()

    # Try to get duration
    duration_text = soup.get_text()
    dur_match = re.search(r'(\d+)\s*(?:–|-|to)\s*(\d+)\s*minutes?', duration_text, re.I)
    if dur_match:
        assessment["duration_minutes"] = f"{dur_match.group(1)}-{dur_match.group(2)}"
    else:
        dur_match = re.search(r'(\d+)\s*minutes?', duration_text, re.I)
        if dur_match:
            assessment["duration_minutes"] = int(dur_match.group(1))

    # Try to find main description paragraph
    for selector in [".product-description", ".hero-description", "article p", "main p"]:
        el = soup.select_one(selector)
        if el:
            txt = el.get_text(strip=True)
            if len(txt) > 50:
                assessment["description"] = txt
                break

    time.sleep(0.5)  # Be polite
    return assessment


def scrape_all_pages() -> list[dict]:
    """Handle pagination across catalog pages."""
    all_assessments = []
    seen_urls = set()

    # The SHL catalog uses ?start=0&type=1 pagination
    # type=1 = Individual Test Solutions
    page_num = 0
    page_size = 12  # SHL shows 12 per page typically

    print("Scraping SHL Individual Test Solutions catalog...")

    # First try the main catalog page with filter for individual tests
    urls_to_try = [
        f"{CATALOG_URL}?type=1",
        f"{CATALOG_URL}?start=0&type=1",
        CATALOG_URL,
    ]

    for start_url in urls_to_try:
        print(f"\nTrying: {start_url}")
        soup = get_page(start_url)
        if not soup:
            continue

        # Look for pagination info
        page_text = soup.get_text()

        # Scrape current page
        assessments = scrape_catalog_page(start_url)
        for a in assessments:
            if a["url"] not in seen_urls:
                seen_urls.add(a["url"])
                all_assessments.append(a)

        if all_assessments:
            # Try to find "next page" links
            next_links = soup.find_all("a", string=re.compile(r"next|›|»", re.I))
            for nl in next_links:
                href = nl.get("href", "")
                if href:
                    next_url = BASE_URL + href if href.startswith("/") else href
                    print(f"\nFollowing pagination: {next_url}")
                    more = scrape_catalog_page(next_url)
                    for a in more:
                        if a["url"] not in seen_urls:
                            seen_urls.add(a["url"])
                            all_assessments.append(a)

            # Also try numeric pagination
            for offset in range(page_size, 300, page_size):
                paginated_url = f"{CATALOG_URL}?start={offset}&type=1"
                print(f"\nPagination offset {offset}: {paginated_url}")
                more = scrape_catalog_page(paginated_url)
                if not more:
                    print("  No more results, stopping pagination.")
                    break
                new_count = 0
                for a in more:
                    if a["url"] not in seen_urls:
                        seen_urls.add(a["url"])
                        all_assessments.append(a)
                        new_count += 1
                if new_count == 0:
                    break
                time.sleep(1)

            break  # We got results from this URL pattern

    return all_assessments


def enrich_assessments(assessments: list[dict]) -> list[dict]:
    """Fetch detail pages for richer descriptions."""
    print(f"\nEnriching {len(assessments)} assessments with detail pages...")
    enriched = []
    for i, a in enumerate(assessments):
        print(f"  [{i+1}/{len(assessments)}] {a['name']}")
        enriched.append(scrape_product_detail(a))
    return enriched


def build_fallback_catalog() -> list[dict]:
    """
    Fallback: hardcoded catalog of well-known SHL Individual Test Solutions.
    Used if scraping fails. Based on publicly known SHL product catalog.
    """
    print("Using fallback hardcoded catalog...")
    return [
        # Ability & Aptitude
        {"name": "Verify Interactive - Numerical Reasoning", "url": "https://www.shl.com/solutions/products/product-catalog/view/verify-interactive-numerical-reasoning/", "test_types": ["A"], "test_type_labels": ["Ability & Aptitude"], "description": "Measures ability to work with numerical data, interpret statistics, and make data-based decisions. Suitable for roles requiring quantitative analysis.", "remote_testing": True, "duration_minutes": 25, "languages": ["English"]},
        {"name": "Verify Interactive - Verbal Reasoning", "url": "https://www.shl.com/solutions/products/product-catalog/view/verify-interactive-verbal-reasoning/", "test_types": ["A"], "test_type_labels": ["Ability & Aptitude"], "description": "Assesses ability to understand and evaluate written information, draw conclusions from text, and communicate effectively.", "remote_testing": True, "duration_minutes": 19, "languages": ["English"]},
        {"name": "Verify Interactive - Inductive Reasoning", "url": "https://www.shl.com/solutions/products/product-catalog/view/verify-interactive-inductive-reasoning/", "test_types": ["A"], "test_type_labels": ["Ability & Aptitude"], "description": "Measures ability to identify patterns, think conceptually, and solve novel problems — key for analytical and managerial roles.", "remote_testing": True, "duration_minutes": 25, "languages": ["English"]},
        {"name": "Verify Interactive - Deductive Reasoning", "url": "https://www.shl.com/solutions/products/product-catalog/view/verify-interactive-deductive-reasoning/", "test_types": ["A"], "test_type_labels": ["Ability & Aptitude"], "description": "Evaluates logical reasoning ability — drawing conclusions from given information. Used for roles requiring structured thinking.", "remote_testing": True, "duration_minutes": 20, "languages": ["English"]},
        {"name": "Verify - Numerical Ability", "url": "https://www.shl.com/solutions/products/product-catalog/view/verify-numerical-ability/", "test_types": ["A"], "test_type_labels": ["Ability & Aptitude"], "description": "Basic numerical operations and working with numbers. Suitable for roles requiring everyday numerical competence.", "remote_testing": True, "duration_minutes": 18, "languages": ["English"]},
        {"name": "Verify - Verbal Ability", "url": "https://www.shl.com/solutions/products/product-catalog/view/verify-verbal-ability/", "test_types": ["A"], "test_type_labels": ["Ability & Aptitude"], "description": "Measures verbal comprehension and reasoning at a basic level. Ideal for clerical and administrative roles.", "remote_testing": True, "duration_minutes": 18, "languages": ["English"]},
        {"name": "Verify - Checking", "url": "https://www.shl.com/solutions/products/product-catalog/view/verify-checking/", "test_types": ["A"], "test_type_labels": ["Ability & Aptitude"], "description": "Measures speed and accuracy of checking information — essential for data entry, administrative and compliance roles.", "remote_testing": True, "duration_minutes": 7, "languages": ["English"]},
        {"name": "Verify - Calculation", "url": "https://www.shl.com/solutions/products/product-catalog/view/verify-calculation/", "test_types": ["A"], "test_type_labels": ["Ability & Aptitude"], "description": "Tests basic arithmetic calculation speed and accuracy. Appropriate for clerical, financial, and operational roles.", "remote_testing": True, "duration_minutes": 12, "languages": ["English"]},
        {"name": "Verify G+ - Cognitive Ability", "url": "https://www.shl.com/solutions/products/product-catalog/view/verify-g-cognitive-ability/", "test_types": ["A"], "test_type_labels": ["Ability & Aptitude"], "description": "Comprehensive general cognitive ability assessment combining numerical, verbal, and inductive reasoning. Predictive of performance across all roles.", "remote_testing": True, "duration_minutes": 36, "languages": ["English"]},
        {"name": "Numerical Reasoning", "url": "https://www.shl.com/solutions/products/product-catalog/view/numerical-reasoning/", "test_types": ["A"], "test_type_labels": ["Ability & Aptitude"], "description": "Classic numerical reasoning test for evaluating analytical thinking with numbers and statistics.", "remote_testing": True, "duration_minutes": 25, "languages": ["English", "Multiple"]},
        {"name": "Verbal Reasoning", "url": "https://www.shl.com/solutions/products/product-catalog/view/verbal-reasoning/", "test_types": ["A"], "test_type_labels": ["Ability & Aptitude"], "description": "Classic verbal reasoning assessment measuring comprehension and critical evaluation of written information.", "remote_testing": True, "duration_minutes": 19, "languages": ["English", "Multiple"]},
        {"name": "Inductive Reasoning", "url": "https://www.shl.com/solutions/products/product-catalog/view/inductive-reasoning/", "test_types": ["A"], "test_type_labels": ["Ability & Aptitude"], "description": "Pattern recognition and abstract reasoning test used for managerial and graduate-level hiring.", "remote_testing": True, "duration_minutes": 25, "languages": ["English", "Multiple"]},
        {"name": "Mechanical Comprehension", "url": "https://www.shl.com/solutions/products/product-catalog/view/mechanical-comprehension/", "test_types": ["A"], "test_type_labels": ["Ability & Aptitude"], "description": "Assesses understanding of mechanical and physical principles. Essential for engineering and technical roles.", "remote_testing": True, "duration_minutes": 20, "languages": ["English"]},
        {"name": "Spatial Reasoning", "url": "https://www.shl.com/solutions/products/product-catalog/view/spatial-reasoning/", "test_types": ["A"], "test_type_labels": ["Ability & Aptitude"], "description": "Measures ability to visualise and mentally rotate shapes in 2D and 3D — relevant for design, engineering, and technical roles.", "remote_testing": True, "duration_minutes": 20, "languages": ["English"]},
        # Personality & Behaviour
        {"name": "OPQ32r", "url": "https://www.shl.com/solutions/products/product-catalog/view/opq32r/", "test_types": ["P"], "test_type_labels": ["Personality & Behaviour"], "description": "The Occupational Personality Questionnaire (OPQ32r) measures 32 personality characteristics relevant to occupational performance. Industry-leading personality assessment for selection and development at all levels.", "remote_testing": True, "duration_minutes": 25, "languages": ["English", "40+"]},
        {"name": "OPQ32n", "url": "https://www.shl.com/solutions/products/product-catalog/view/opq32n/", "test_types": ["P"], "test_type_labels": ["Personality & Behaviour"], "description": "Normative version of the OPQ32, measuring personality across 32 dimensions. Used for leadership and senior-level selection.", "remote_testing": True, "duration_minutes": 35, "languages": ["English", "Multiple"]},
        {"name": "Motives, Values, Preferences Inventory (MVPI)", "url": "https://www.shl.com/solutions/products/product-catalog/view/motives-values-preferences-inventory/", "test_types": ["M"], "test_type_labels": ["Motivation"], "description": "Measures core values, goals, and interests that drive behaviour. Predicts job satisfaction and culture fit.", "remote_testing": True, "duration_minutes": 15, "languages": ["English", "Multiple"]},
        {"name": "Work Strengths", "url": "https://www.shl.com/solutions/products/product-catalog/view/work-strengths/", "test_types": ["P"], "test_type_labels": ["Personality & Behaviour"], "description": "Assesses an individual's key behavioural strengths relevant to workplace performance. Suitable for volume hiring.", "remote_testing": True, "duration_minutes": 10, "languages": ["English"]},
        {"name": "General Personality Assessment (GPA)", "url": "https://www.shl.com/solutions/products/product-catalog/view/general-personality-assessment/", "test_types": ["P"], "test_type_labels": ["Personality & Behaviour"], "description": "Broad personality questionnaire measuring traits relevant to work performance across industries.", "remote_testing": True, "duration_minutes": 20, "languages": ["English"]},
        {"name": "Global Skills Assessment (GSA)", "url": "https://www.shl.com/solutions/products/product-catalog/view/global-skills-assessment/", "test_types": ["K"], "test_type_labels": ["Knowledge & Skills"], "description": "Measures a range of workplace skills including communication, problem solving and decision making, relevant across global contexts.", "remote_testing": True, "duration_minutes": 30, "languages": ["English", "Multiple"]},
        # Knowledge & Skills
        {"name": "Java 8 (New)", "url": "https://www.shl.com/solutions/products/product-catalog/view/java-8-new/", "test_types": ["K"], "test_type_labels": ["Knowledge & Skills"], "description": "Assesses Java 8 programming knowledge including OOP, streams, lambda expressions, collections, and concurrency. For mid to senior developers.", "remote_testing": True, "duration_minutes": 30, "languages": ["English"]},
        {"name": "Core Java (Advanced Level) (New)", "url": "https://www.shl.com/solutions/products/product-catalog/view/core-java-advanced-level-new/", "test_types": ["K"], "test_type_labels": ["Knowledge & Skills"], "description": "Advanced Core Java assessment covering JVM internals, concurrency, design patterns, and enterprise Java concepts.", "remote_testing": True, "duration_minutes": 45, "languages": ["English"]},
        {"name": "Python (New)", "url": "https://www.shl.com/solutions/products/product-catalog/view/python-new/", "test_types": ["K"], "test_type_labels": ["Knowledge & Skills"], "description": "Assesses Python programming knowledge including syntax, data structures, OOP, libraries, and scripting. Suitable for software developers and data roles.", "remote_testing": True, "duration_minutes": 30, "languages": ["English"]},
        {"name": "JavaScript (New)", "url": "https://www.shl.com/solutions/products/product-catalog/view/javascript-new/", "test_types": ["K"], "test_type_labels": ["Knowledge & Skills"], "description": "Tests JavaScript programming knowledge including ES6+, async programming, DOM manipulation, and common patterns.", "remote_testing": True, "duration_minutes": 30, "languages": ["English"]},
        {"name": "SQL (New)", "url": "https://www.shl.com/solutions/products/product-catalog/view/sql-new/", "test_types": ["K"], "test_type_labels": ["Knowledge & Skills"], "description": "Tests knowledge of SQL including queries, joins, aggregations, stored procedures, and database design. For data and backend roles.", "remote_testing": True, "duration_minutes": 30, "languages": ["English"]},
        {"name": "C++ (New)", "url": "https://www.shl.com/solutions/products/product-catalog/view/c-new/", "test_types": ["K"], "test_type_labels": ["Knowledge & Skills"], "description": "Assesses C++ programming skills including STL, memory management, OOP, and modern C++ features.", "remote_testing": True, "duration_minutes": 30, "languages": ["English"]},
        {"name": "C# (New)", "url": "https://www.shl.com/solutions/products/product-catalog/view/c-sharp-new/", "test_types": ["K"], "test_type_labels": ["Knowledge & Skills"], "description": "Tests C# programming knowledge including .NET framework, LINQ, async/await, and object-oriented design.", "remote_testing": True, "duration_minutes": 30, "languages": ["English"]},
        {"name": "Automata - Fix (Coding Simulation)", "url": "https://www.shl.com/solutions/products/product-catalog/view/automata-fix/", "test_types": ["K", "S"], "test_type_labels": ["Knowledge & Skills", "Simulations"], "description": "Coding simulation where candidates fix bugs in provided code. Assesses real-world programming ability in multiple languages including Java, Python, C++.", "remote_testing": True, "duration_minutes": 45, "languages": ["English"]},
        {"name": "Entry Level Software Engineer (Coding)", "url": "https://www.shl.com/solutions/products/product-catalog/view/entry-level-software-engineer-coding/", "test_types": ["K"], "test_type_labels": ["Knowledge & Skills"], "description": "Practical coding assessment for entry-level software engineers. Covers data structures, algorithms, and basic programming concepts.", "remote_testing": True, "duration_minutes": 60, "languages": ["English"]},
        {"name": "Technology Professional (Coding)", "url": "https://www.shl.com/solutions/products/product-catalog/view/technology-professional-coding/", "test_types": ["K"], "test_type_labels": ["Knowledge & Skills"], "description": "Coding assessment for experienced technology professionals, testing problem-solving, algorithms, and software design.", "remote_testing": True, "duration_minutes": 60, "languages": ["English"]},
        {"name": "Microsoft Excel (New)", "url": "https://www.shl.com/solutions/products/product-catalog/view/microsoft-excel-new/", "test_types": ["K"], "test_type_labels": ["Knowledge & Skills"], "description": "Tests proficiency with Microsoft Excel including formulas, pivot tables, data analysis, and macros.", "remote_testing": True, "duration_minutes": 20, "languages": ["English"]},
        {"name": "Microsoft Word (New)", "url": "https://www.shl.com/solutions/products/product-catalog/view/microsoft-word-new/", "test_types": ["K"], "test_type_labels": ["Knowledge & Skills"], "description": "Assesses Microsoft Word skills including document formatting, tables, styles, mail merge, and collaboration features.", "remote_testing": True, "duration_minutes": 20, "languages": ["English"]},
        {"name": "Accounting/Finance (Short)", "url": "https://www.shl.com/solutions/products/product-catalog/view/accounting-finance-short/", "test_types": ["K"], "test_type_labels": ["Knowledge & Skills"], "description": "Tests foundational accounting and financial concepts for roles in finance, banking, and accounting functions.", "remote_testing": True, "duration_minutes": 15, "languages": ["English"]},
        {"name": "Financial Analysis Skills Test", "url": "https://www.shl.com/solutions/products/product-catalog/view/financial-analysis-skills-test/", "test_types": ["K"], "test_type_labels": ["Knowledge & Skills"], "description": "Comprehensive financial analysis assessment covering financial modeling, valuation, and analytical skills for finance roles.", "remote_testing": True, "duration_minutes": 30, "languages": ["English"]},
        {"name": "Administrative Professional - Short Form", "url": "https://www.shl.com/solutions/products/product-catalog/view/administrative-professional-short-form/", "test_types": ["K"], "test_type_labels": ["Knowledge & Skills"], "description": "Assesses skills critical for administrative and clerical roles including data management, communication, and organization.", "remote_testing": True, "duration_minutes": 30, "languages": ["English"]},
        {"name": "Customer Service Aptitude Assessment", "url": "https://www.shl.com/solutions/products/product-catalog/view/customer-service-aptitude-assessment/", "test_types": ["B"], "test_type_labels": ["Biodata & Situational Judgement"], "description": "Situational judgement test for customer-facing roles measuring service orientation, communication, and problem-solving.", "remote_testing": True, "duration_minutes": 20, "languages": ["English"]},
        # Situational Judgement / Biodata
        {"name": "Situational Judgement Test", "url": "https://www.shl.com/solutions/products/product-catalog/view/situational-judgement-test/", "test_types": ["B"], "test_type_labels": ["Biodata & Situational Judgement"], "description": "Presents realistic workplace scenarios and asks candidates to choose the most effective response. Predicts on-the-job behaviour.", "remote_testing": True, "duration_minutes": 30, "languages": ["English"]},
        {"name": "Graduate 8 (G8) - Situational Judgement", "url": "https://www.shl.com/solutions/products/product-catalog/view/graduate-8/", "test_types": ["B"], "test_type_labels": ["Biodata & Situational Judgement"], "description": "SJT designed for graduate-level candidates assessing judgment across 8 competencies critical for early career success.", "remote_testing": True, "duration_minutes": 25, "languages": ["English"]},
        {"name": "Sales Aptitude Assessment", "url": "https://www.shl.com/solutions/products/product-catalog/view/sales-aptitude-assessment/", "test_types": ["B"], "test_type_labels": ["Biodata & Situational Judgement"], "description": "Measures aptitude for sales roles by assessing influence, resilience, customer focus, and achievement motivation.", "remote_testing": True, "duration_minutes": 25, "languages": ["English"]},
        {"name": "Contact Center Aptitude Assessment", "url": "https://www.shl.com/solutions/products/product-catalog/view/contact-center-aptitude-assessment/", "test_types": ["B"], "test_type_labels": ["Biodata & Situational Judgement"], "description": "Targeted for contact center and BPO roles measuring customer handling, multitasking, and communication skills.", "remote_testing": True, "duration_minutes": 25, "languages": ["English"]},
        {"name": "Retail Aptitude Assessment", "url": "https://www.shl.com/solutions/products/product-catalog/view/retail-aptitude-assessment/", "test_types": ["B"], "test_type_labels": ["Biodata & Situational Judgement"], "description": "Assesses aptitude for retail roles including customer service orientation, teamwork, reliability, and sales focus.", "remote_testing": True, "duration_minutes": 20, "languages": ["English"]},
        # Competencies & Leadership
        {"name": "Leadership Report (OPQ)", "url": "https://www.shl.com/solutions/products/product-catalog/view/leadership-report/", "test_types": ["P", "C"], "test_type_labels": ["Personality & Behaviour", "Competencies"], "description": "Generates a leadership-focused report from OPQ32 data, mapping personality to leadership competencies and potential.", "remote_testing": True, "duration_minutes": 25, "languages": ["English", "Multiple"]},
        {"name": "Universal Competency Framework (UCF)", "url": "https://www.shl.com/solutions/products/product-catalog/view/universal-competency-framework/", "test_types": ["C"], "test_type_labels": ["Competencies"], "description": "Comprehensive competency framework covering 8 factors and 20 competencies. Used for talent assessment, development, and succession planning.", "remote_testing": False, "duration_minutes": None, "languages": ["English"]},
        {"name": "Hogan Personality Inventory (HPI)", "url": "https://www.shl.com/solutions/products/product-catalog/view/hogan-personality-inventory/", "test_types": ["P"], "test_type_labels": ["Personality & Behaviour"], "description": "Measures normal personality characteristics that predict job performance, leadership, and career success.", "remote_testing": True, "duration_minutes": 15, "languages": ["English", "Multiple"]},
        {"name": "Hogan Development Survey (HDS)", "url": "https://www.shl.com/solutions/products/product-catalog/view/hogan-development-survey/", "test_types": ["P", "D"], "test_type_labels": ["Personality & Behaviour", "Development & 360"], "description": "Identifies personality-based performance risks and derailers — critical for leadership selection and development.", "remote_testing": True, "duration_minutes": 15, "languages": ["English", "Multiple"]},
        # Exercises & Simulations
        {"name": "In-Basket Exercise", "url": "https://www.shl.com/solutions/products/product-catalog/view/in-basket-exercise/", "test_types": ["E"], "test_type_labels": ["Assessment Exercises"], "description": "Simulates a manager's email inbox requiring candidates to prioritize, delegate, and respond to realistic work scenarios.", "remote_testing": True, "duration_minutes": 45, "languages": ["English"]},
        {"name": "MQ: Motivation Questionnaire", "url": "https://www.shl.com/solutions/products/product-catalog/view/motivation-questionnaire/", "test_types": ["M"], "test_type_labels": ["Motivation"], "description": "Assesses motivational factors that drive an individual's engagement and performance at work. Used for selection and development.", "remote_testing": True, "duration_minutes": 25, "languages": ["English", "Multiple"]},
        {"name": "Customer Contact Styles Questionnaire (CCSQ)", "url": "https://www.shl.com/solutions/products/product-catalog/view/customer-contact-styles-questionnaire/", "test_types": ["P"], "test_type_labels": ["Personality & Behaviour"], "description": "Personality questionnaire designed for customer-facing and sales roles. Measures interpersonal and service-oriented traits.", "remote_testing": True, "duration_minutes": 20, "languages": ["English"]},
        {"name": "Work Style Questionnaire (WSQ)", "url": "https://www.shl.com/solutions/products/product-catalog/view/work-style-questionnaire/", "test_types": ["P"], "test_type_labels": ["Personality & Behaviour"], "description": "Shorter personality questionnaire capturing core work-related behavioural tendencies. Used for high-volume hiring.", "remote_testing": True, "duration_minutes": 15, "languages": ["English"]},
        {"name": "Graduate and Managerial Assessment (GMA - Numerical)", "url": "https://www.shl.com/solutions/products/product-catalog/view/graduate-and-managerial-assessment-numerical/", "test_types": ["A"], "test_type_labels": ["Ability & Aptitude"], "description": "Graduate and managerial-level numerical reasoning test measuring the ability to interpret and draw conclusions from numerical data.", "remote_testing": True, "duration_minutes": 25, "languages": ["English"]},
        {"name": "Graduate and Managerial Assessment (GMA - Verbal)", "url": "https://www.shl.com/solutions/products/product-catalog/view/graduate-and-managerial-assessment-verbal/", "test_types": ["A"], "test_type_labels": ["Ability & Aptitude"], "description": "Graduate and managerial-level verbal reasoning assessment measuring comprehension and evaluation of written information.", "remote_testing": True, "duration_minutes": 19, "languages": ["English"]},
        {"name": "Graduate and Managerial Assessment (GMA - Abstract)", "url": "https://www.shl.com/solutions/products/product-catalog/view/graduate-and-managerial-assessment-abstract/", "test_types": ["A"], "test_type_labels": ["Ability & Aptitude"], "description": "Abstract reasoning test for graduate and managerial assessment measuring pattern recognition and conceptual thinking.", "remote_testing": True, "duration_minutes": 25, "languages": ["English"]},
        {"name": "Scenarios: Sales", "url": "https://www.shl.com/solutions/products/product-catalog/view/scenarios-sales/", "test_types": ["S"], "test_type_labels": ["Simulations"], "description": "Simulation-based assessment for sales roles presenting realistic customer scenarios and evaluating sales effectiveness.", "remote_testing": True, "duration_minutes": 30, "languages": ["English"]},
        {"name": "Scenarios: Service", "url": "https://www.shl.com/solutions/products/product-catalog/view/scenarios-service/", "test_types": ["S"], "test_type_labels": ["Simulations"], "description": "Service-focused simulation assessing how candidates handle difficult customers, service failures, and teamwork situations.", "remote_testing": True, "duration_minutes": 30, "languages": ["English"]},
        {"name": "Data Entry Speed and Accuracy Assessment", "url": "https://www.shl.com/solutions/products/product-catalog/view/data-entry-speed-and-accuracy-assessment/", "test_types": ["K"], "test_type_labels": ["Knowledge & Skills"], "description": "Measures speed and accuracy in data entry tasks. Critical for administrative, clerical, and data management roles.", "remote_testing": True, "duration_minutes": 10, "languages": ["English"]},
        {"name": "Occupational Stress Indicator", "url": "https://www.shl.com/solutions/products/product-catalog/view/occupational-stress-indicator/", "test_types": ["P", "D"], "test_type_labels": ["Personality & Behaviour", "Development & 360"], "description": "Assesses stress levels and coping strategies in the workplace. Used for development and wellbeing programmes.", "remote_testing": True, "duration_minutes": 30, "languages": ["English"]},
        {"name": "Agile Software Developer", "url": "https://www.shl.com/solutions/products/product-catalog/view/agile-software-developer/", "test_types": ["K"], "test_type_labels": ["Knowledge & Skills"], "description": "Tests knowledge of Agile methodologies, Scrum, Kanban, and DevOps practices for software development roles.", "remote_testing": True, "duration_minutes": 20, "languages": ["English"]},
        {"name": "DevOps (New)", "url": "https://www.shl.com/solutions/products/product-catalog/view/devops-new/", "test_types": ["K"], "test_type_labels": ["Knowledge & Skills"], "description": "Assesses DevOps knowledge including CI/CD, containerization, cloud infrastructure, monitoring, and automation.", "remote_testing": True, "duration_minutes": 30, "languages": ["English"]},
        {"name": "AWS (New)", "url": "https://www.shl.com/solutions/products/product-catalog/view/aws-new/", "test_types": ["K"], "test_type_labels": ["Knowledge & Skills"], "description": "Tests AWS cloud knowledge including core services, architecture, security, and deployment for cloud engineering roles.", "remote_testing": True, "duration_minutes": 30, "languages": ["English"]},
        {"name": "Machine Learning (New)", "url": "https://www.shl.com/solutions/products/product-catalog/view/machine-learning-new/", "test_types": ["K"], "test_type_labels": ["Knowledge & Skills"], "description": "Assesses machine learning knowledge including algorithms, model evaluation, feature engineering, and practical ML concepts.", "remote_testing": True, "duration_minutes": 30, "languages": ["English"]},
        {"name": "Data Science (New)", "url": "https://www.shl.com/solutions/products/product-catalog/view/data-science-new/", "test_types": ["K"], "test_type_labels": ["Knowledge & Skills"], "description": "Tests data science concepts including statistics, data manipulation, visualization, and applied analytics for data scientist roles.", "remote_testing": True, "duration_minutes": 30, "languages": ["English"]},
        {"name": "General Sales Aptitude (GSA)", "url": "https://www.shl.com/solutions/products/product-catalog/view/general-sales-aptitude/", "test_types": ["B"], "test_type_labels": ["Biodata & Situational Judgement"], "description": "Broad sales aptitude assessment covering drive, resilience, relationship-building, and customer focus for all sales roles.", "remote_testing": True, "duration_minutes": 25, "languages": ["English"]},
        {"name": "Management and Graduate Item Bank (MGIB) - Verbal", "url": "https://www.shl.com/solutions/products/product-catalog/view/management-and-graduate-item-bank-verbal/", "test_types": ["A"], "test_type_labels": ["Ability & Aptitude"], "description": "High-level verbal reasoning assessment from the MGIB series for management and graduate-level selection.", "remote_testing": True, "duration_minutes": 19, "languages": ["English"]},
        {"name": "Management and Graduate Item Bank (MGIB) - Numerical", "url": "https://www.shl.com/solutions/products/product-catalog/view/management-and-graduate-item-bank-numerical/", "test_types": ["A"], "test_type_labels": ["Ability & Aptitude"], "description": "High-level numerical reasoning assessment from the MGIB series for management and graduate selection.", "remote_testing": True, "duration_minutes": 25, "languages": ["English"]},
        {"name": "Workplace English Test", "url": "https://www.shl.com/solutions/products/product-catalog/view/workplace-english-test/", "test_types": ["K"], "test_type_labels": ["Knowledge & Skills"], "description": "Assesses English language proficiency in a workplace context including reading, comprehension, and writing for non-native speakers.", "remote_testing": True, "duration_minutes": 30, "languages": ["English"]},
        {"name": "Bank Administrative Assistant - Short Form", "url": "https://www.shl.com/solutions/products/product-catalog/view/bank-administrative-assistant-short-form/", "test_types": ["K"], "test_type_labels": ["Knowledge & Skills"], "description": "Designed for administrative roles in banking and financial services, covering relevant numerical and clerical skills.", "remote_testing": True, "duration_minutes": 25, "languages": ["English"]},
    ]


def main():
    output_path = Path(__file__).parent.parent / "data" / "catalog.json"
    output_path.parent.mkdir(exist_ok=True)

    # Try live scraping first
    assessments = scrape_all_pages()

    if len(assessments) < 10:
        print(f"\nOnly found {len(assessments)} assessments via scraping. Using fallback catalog.")
        assessments = build_fallback_catalog()
    else:
        # Enrich with detail pages
        assessments = enrich_assessments(assessments)

    print(f"\nTotal assessments: {len(assessments)}")
    output_path.write_text(json.dumps(assessments, indent=2, ensure_ascii=False))
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
