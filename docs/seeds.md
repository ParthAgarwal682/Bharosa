# Bharosa Verified Crawler Seeds Documentation

This document records the verified official Government of India sources configured in `config/seeds.yaml` for Bharosa's live crawl corpus.

## Source Verification & Provenance

### 1. Atal Vayo Abhyuday Yojana (AVYAY)
- **URL:** `https://socialjustice.gov.in/schemes/43`
- **Official Organization:** Department of Social Justice and Empowerment, Ministry of Social Justice and Empowerment, Government of India
- **Domain:** `socialjustice.gov.in`
- **Why Selected:** Central flagship scheme for health and welfare of indigent senior citizens, including assisted living devices, medical healthcare support, and shelter care. Directly serves families who cannot afford healthcare.
- **HTTP Status:** 200 OK (Content-Length: ~65 KB)
- **Final URL:** `https://socialjustice.gov.in/schemes/43`
- **robots.txt:** Allowed (`User-agent: * Allow: /` with sitemap reference)
- **Terms/Restrictions:** Official public portal for government welfare schemes; automated polite crawling permitted.
- **Crawler Compatibility:** Static server-rendered HTML; extracts cleanly into 8 structured sections in `ZoneDoc` pipeline.

### 2. PM Special — Training of Geriatric Care Givers
- **URL:** `https://socialjustice.gov.in/schemes/111`
- **Official Organization:** Department of Social Justice and Empowerment, Ministry of Social Justice and Empowerment, Government of India
- **Domain:** `socialjustice.gov.in`
- **Why Selected:** Specialized government healthcare assistance scheme training bedside caregivers for elderly patients and chronic illnesses.
- **HTTP Status:** 200 OK (Content-Length: ~60 KB)
- **Final URL:** `https://socialjustice.gov.in/schemes/111`
- **robots.txt:** Allowed
- **Terms/Restrictions:** Public government scheme portal; no crawling restrictions.
- **Crawler Compatibility:** Static server-rendered HTML; extracts cleanly into 8 structured sections.

### 3. National Action Plan for Drug Demand Reduction
- **URL:** `https://socialjustice.gov.in/schemes/42`
- **Official Organization:** Department of Social Justice and Empowerment, Ministry of Social Justice and Empowerment, Government of India
- **Domain:** `socialjustice.gov.in`
- **Why Selected:** Healthcare, medical rehabilitation, and de-addiction counselling scheme providing clinical facilities and community healthcare.
- **HTTP Status:** 200 OK (Content-Length: ~70 KB)
- **Final URL:** `https://socialjustice.gov.in/schemes/42`
- **robots.txt:** Allowed
- **Terms/Restrictions:** Public government scheme portal; no crawling restrictions.
- **Crawler Compatibility:** Static server-rendered HTML; extracts cleanly into 7 structured sections.

### 4. NAMASTE Scheme
- **URL:** `https://socialjustice.gov.in/schemes/37`
- **Official Organization:** Department of Social Justice and Empowerment, Ministry of Social Justice and Empowerment, Government of India
- **Domain:** `socialjustice.gov.in`
- **Why Selected:** National Action for Mechanised Sanitation Ecosystem; provides health insurance convergence under Ayushman Bharat PM-JAY, health checkups, and safety gear for sanitation workers.
- **HTTP Status:** 200 OK (Content-Length: ~62 KB)
- **Final URL:** `https://socialjustice.gov.in/schemes/37`
- **robots.txt:** Allowed
- **Terms/Restrictions:** Public government scheme portal; no crawling restrictions.
- **Crawler Compatibility:** Static server-rendered HTML; extracts cleanly into 7 structured sections.

### 5. SMILE Scheme
- **URL:** `https://socialjustice.gov.in/schemes/99`
- **Official Organization:** Department of Social Justice and Empowerment, Ministry of Social Justice and Empowerment, Government of India
- **Domain:** `socialjustice.gov.in`
- **Why Selected:** Support for Marginalized Individuals for Livelihood and Enterprise; provides comprehensive medical care, psychiatric care, and rehabilitation for destitute individuals.
- **HTTP Status:** 200 OK (Content-Length: ~58 KB)
- **Final URL:** `https://socialjustice.gov.in/schemes/99`
- **robots.txt:** Allowed
- **Terms/Restrictions:** Public government scheme portal; no crawling restrictions.
- **Crawler Compatibility:** Static server-rendered HTML; extracts cleanly into 7 structured sections.

### 6. Mission Saksham Anganwadi & POSHAN 2.0
- **URL:** `https://wcd.gov.in/offerings/schemes-services-mission-poshan-2-0`
- **Official Organization:** Ministry of Women and Child Development, Government of India
- **Domain:** `wcd.gov.in`
- **Why Selected:** Flagship national nutrition and healthcare mission addressing maternal malnutrition, infant healthcare, and lactating mothers' wellness.
- **HTTP Status:** 200 OK (Content-Length: ~53 KB)
- **Final URL:** `https://wcd.gov.in/offerings/schemes-services-mission-poshan-2-0`
- **robots.txt:** Allowed (`User-agent: * Allow: /`)
- **Terms/Restrictions:** Public government portal; standard respectful access permitted.
- **Crawler Compatibility:** Static server-rendered HTML; extracts cleanly into 5 structured sections.

### 7. Mission Vatsalya
- **URL:** `https://wcd.gov.in/offerings/mission-vatsalya`
- **Official Organization:** Ministry of Women and Child Development, Government of India
- **Domain:** `wcd.gov.in`
- **Why Selected:** Child welfare, healthcare access, survival, and protection scheme supporting vulnerable children and families in distress.
- **HTTP Status:** 200 OK (Content-Length: ~54 KB)
- **Final URL:** `https://wcd.gov.in/offerings/mission-vatsalya`
- **robots.txt:** Allowed
- **Terms/Restrictions:** Public government portal; no crawling restrictions.
- **Crawler Compatibility:** Static server-rendered HTML; extracts cleanly into 5 structured sections.

### 8. Mission Shakti
- **URL:** `https://wcd.gov.in/offerings/mission-shakti-guidelines-for-implementation`
- **Official Organization:** Ministry of Women and Child Development, Government of India
- **Domain:** `wcd.gov.in`
- **Why Selected:** Umbrella scheme for women's healthcare, shelter, legal and psychological support, and medical crisis intervention.
- **HTTP Status:** 200 OK (Content-Length: ~53 KB)
- **Final URL:** `https://wcd.gov.in/offerings/mission-shakti-guidelines-for-implementation`
- **robots.txt:** Allowed
- **Terms/Restrictions:** Public government portal; no crawling restrictions.
- **Crawler Compatibility:** Static server-rendered HTML; extracts cleanly into 5 structured sections.

### 9. Technical Divisions of DGHS
- **URL:** `https://dghs.mohfw.gov.in/technical-divisions-of-dghs.php`
- **Official Organization:** Directorate General of Health Services, Ministry of Health and Family Welfare, Government of India
- **Domain:** `dghs.mohfw.gov.in`
- **Why Selected:** Apex technical health authority detailing national health programmes, leprosy eradication, non-communicable disease (NCD) control, and nutritional programmes.
- **HTTP Status:** 200 OK (Content-Length: ~26 KB)
- **Final URL:** `https://dghs.mohfw.gov.in/technical-divisions-of-dghs.php`
- **robots.txt:** Allowed (No disallow rules)
- **Terms/Restrictions:** Official Directorate General of Health Services portal; public access.
- **Crawler Compatibility:** Server-rendered HTML; extracts into 4 structured sections.
