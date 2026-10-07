# Medicine Dataset Audit (DATA_AUDIT.md)

**Audit Date:** October 2026  
**Auditor / Role:** Module B — Medicine Data & Baselines  
**Dataset Target:** `data/medicines/indian_pharmaceutical_products_clean.csv`  
**Loader Implementation:** `bharosa/medicine/loader.py`

---

## 1. Provenance & Medical Authority Disclaimer

> [!WARNING]
> **NO MEDICAL AUTHORITY CLAIMED:** This dataset and its loader output are for computational information retrieval (IR) experimentation and evaluation only. They do not constitute clinical guidance, prescribing advice, or verified medical equivalence. All medicine substitutions must be confirmed with a licensed doctor or pharmacist.

- **Source Provenance:** `VERIFY_REQUIRED` (File provided statically in repository; external upstream license and official origin URL require team verification).
- **Dataset License:** `VERIFY_REQUIRED`.
- **Intended Purpose:** Ingestion and normalization into the canonical `MedicineRecord` schema for deterministic same-salt, same-strength, same-form retrieval without LLM hallucination.

---

## 2. File Specifications & Schema

- **Source Filename:** `data/medicines/indian_pharmaceutical_products_clean.csv`
- **Total Dataset Rows:** `253,973` data rows (`253,974` lines including header).
- **Actual Source Columns (15 total):**
  1. `product_id`: Unique identifier string.
  2. `brand_name`: Commercial product/brand name.
  3. `manufacturer`: Pharmaceutical manufacturing/marketing entity.
  4. `price_inr`: Reported Maximum Retail Price (full-pack MRP).
  5. `is_discontinued`: Boolean string (`True` / `False`).
  6. `dosage_form`: Delivery formulation (e.g., `tablet`, `capsule`, `syrup`).
  7. `pack_size`: Numerical count or volume of the pack (e.g., `10.0`, `100.0`).
  8. `pack_unit`: Packaging unit (e.g., `strip`, `bottle`, `vial`).
  9. `num_active_ingredients`: Count of active pharmaceutical ingredients (integer).
  10. `primary_ingredient`: Primary active salt name.
  11. `primary_strength`: Primary active potency string (e.g., `500mg`, `30mg/5ml`).
  12. `active_ingredients`: Structured list of all ingredient dictionaries (`name`, `strength`, `full_description`).
  13. `therapeutic_class`: Broad clinical category (e.g., `antibiotic`, `analgesic`).
  14. `packaging_raw`: Unparsed packaging description string.
  15. `manufacturer_raw`: Unparsed manufacturer name string.

---

## 3. Loader Ingestion & Production Accounting

Execution of `load_medicines("data/medicines/indian_pharmaceutical_products_clean.csv")` produces the following exact accounting:

| Ingestion Metric | Exact Count | Percentage | Description |
|---|---|---|---|
| **Total Rows Processed** | **253,973** | 100.00% | Full dataset row count |
| **Accepted Records** | **230,234** | 90.65% | Valid, production-ready `MedicineRecord` instances |
| **Dropped Rows** | **23,739** | 9.35% | Excluded rows (discontinued, invalid, or duplicate) |
| **Duplicate Counts** | **1,820** | 0.72% | Later duplicates discarded during deduplication |

### Exact Categorized Drop Reasons (`reason_counts`)

| Reason Key | Exact Count | Description & Justification |
|---|---|---|
| `unparseable_strength` | **14,010** | Single-ingredient records where `primary_strength` lacked a positive scalar value and valid unit (e.g. non-numeric strings, irregular formats). Missing strengths are never invented. |
| `discontinued` | **7,905** | Records flagged with `is_discontinued=True`. Excluded from accepted production records to prevent recommending unavailable medicines. |
| `duplicate_record` | **1,820** | Subsequent records with identical `(brand, salt, strength_value, strength_unit, form, manufacturer)` keys. The first occurrence is preserved; subsequent copies are dropped. |
| `invalid_price` | **4** | Records with non-positive ($\le 0$) or non-numeric `price_inr` values. |
| **Total Dropped Check** | **23,739** | Sum matches exact dropped row count ($14{,}010 + 7{,}905 + 1{,}820 + 4 = 23{,}739$). |

---

## 4. Field-by-Field Quality & Variation Analysis

### 4.1 Missingness & Field Coverage
- `brand_name`: 100% populated across valid rows.
- `dosage_form`: 100% populated.
- `manufacturer`: 100% populated across valid rows.
- `price_inr`: Populated across almost all rows (only 4 rows dropped due to non-positive or corrupted prices).
- `primary_ingredient`: Present across all valid single-ingredient records.
- `active_ingredients`: Present as serialized Python/JSON list structures.

### 4.2 Strength Format Issues
- Formats successfully parsed: `650mg`, `650 mg`, `650 MG`, `0.5ml`, `100mcg`, `30mg/5ml`, `50000 IU`, `2%`.
- Unparseable issues (14,010 rows): Includes records with missing strength values, combination strengths in single-ingredient slots without distinct units, or descriptive non-numeric annotations. Per contract, missing strengths are dropped rather than guessed.

### 4.3 Dosage Form Variation
- The source dataset contains diverse casing and plural forms (`tab`, `tabs`, `tablet`, `cap`, `capsule`, `syp`, `syrup`, `inj`, `drops`, `ointment`, `gel`).
- The loader canonicalizes standard aliases to singular base forms (`tab`/`tabs` $\to$ `tablet`, `cap`/`caps` $\to$ `capsule`, `syp` $\to$ `syrup`) to ensure consistent indexing.

### 4.4 Salt & Active-Ingredient Naming
- Contains both single-ingredient medicines (`num_active_ingredients == 1`) and complex multi-salt products (`num_active_ingredients > 1`).
- Multi-salt combination products exhibit varying ingredient ordering in raw strings.

### 4.5 Price Availability
- Source provides `price_inr` representing the price for the declared packaging unit (`pack_size`).
- Minimum valid price: positive decimal; maximum valid price spans high-cost specialty formulations.
- **`generic_price`:** Completely absent from this dataset.

---

## 5. Combination Medicine Handling

To uphold the core safety rule—*never treat combination medicines as interchangeable with single-salt medicines*:
1. **Detection:** Products with `num_active_ingredients > 1` are identified as combinations.
2. **Flagging:** Explicitly marked with `is_combination = True`.
3. **Salt Representation:** The salt is represented as the case-insensitively sorted, joined active ingredient names (`" + ".join(...)`, e.g., `"Amoxycillin + Clavulanic Acid"`). This prevents accidental retrieval when a user searches for a pure single-salt medicine.
4. **No Hallucinated Combined Strength:** For combination products, `strength_value = None` and `strength_unit = None`. The loader strictly refuses to invent an artificial aggregate number (e.g. adding 500mg + 125mg to 625mg is chemically inaccurate and misleading).

---

## 6. Known Limitations

1. **`generic_price` is Unavailable:** The dataset contains brand retail pricing (`price_inr`), but does **not** include Jan Aushadhi (PMBJP) or generic reference prices. In `MedicineRecord`, `generic_price` is strictly set to `None`. It must not be fabricated or assumed to be `0.0`.
2. **Full-Pack Price vs. Unit Price:** `mrp` preserves `price_inr` as the full-pack price (e.g., ₹223.42 for a strip of 10). Downstream comparison algorithms must account for pack sizes if per-unit (per-tablet) price comparisons are required.
3. **Static Snapshot:** The dataset represents a static snapshot and cannot reflect real-time market price revisions, supply shortages, or new regulatory price caps (NPPA).
4. **Exclusion of Discontinued Products:** While discontinued products (`7,905` rows) are tracked in audit statistics, they are dropped from production records to prevent displaying unavailable medicines to users.
