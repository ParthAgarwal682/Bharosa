/**
 * Bharosa Frontend
 * Plain JavaScript interface logic.
 * Handles tab navigation, sample chip population, file uploads, request timeouts, and error handling.
 *
 * Endpoints called:
 * - Medicine: GET /api/medicine/search?q=<query>
 * - Schemes: GET /api/schemes/search?q=<query>
 * - Claim Text Extraction: POST /api/claims/extract-text
 * - Claim Checker: POST /api/claims/check
 */

const REQUEST_TIMEOUT_MS = 120000;
const MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024; // 10MB
const MSG_UNAVAILABLE = 'Search is unavailable right now. Please try again later.';
const MSG_READ_ERROR = 'Something went wrong reading the results.';

document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  initSampleChips();
  initFileUpload();
  initSearchHandlers();
});

function initTabs() {
  const tabs = document.querySelectorAll('.tab-link');
  const panels = document.querySelectorAll('.mode-panel');

  tabs.forEach((tab) => {
    tab.addEventListener('click', () => {
      const targetId = tab.getAttribute('data-target');

      tabs.forEach((t) => {
        t.classList.remove('active');
        t.setAttribute('aria-selected', 'false');
      });
      tab.classList.add('active');
      tab.setAttribute('aria-selected', 'true');

      panels.forEach((panel) => {
        if (panel.id === targetId) {
          panel.classList.add('active');
        } else {
          panel.classList.remove('active');
        }
      });
    });
  });
}

function initSampleChips() {
  const chips = document.querySelectorAll('.sample-chip, .chip');
  chips.forEach((chip) => {
    chip.addEventListener('click', () => {
      const query = chip.getAttribute('data-query') || chip.getAttribute('data-claim');
      if (!query) return;

      const panel = chip.closest('.mode-panel');
      if (panel) {
        const input = panel.querySelector('input[type="text"], textarea');
        if (input) {
          input.value = query;
          input.focus();
        }
      }
    });
  });
}

/**
 * File upload handler supporting .txt and .pdf files.
 * Extracts text and loads it directly into the claim textarea for editing before verification.
 */
function initFileUpload() {
  const btnUpload = document.getElementById('btn-upload-file');
  const fileInput = document.getElementById('claim-file-input');
  const fileStatus = document.getElementById('claim-file-status');
  const fileNameSpan = document.getElementById('claim-file-name');
  const fileError = document.getElementById('claim-file-error');
  const btnClear = document.getElementById('btn-clear-file');
  const claimInput = document.getElementById('input-claim');

  if (!btnUpload || !fileInput) return;

  btnUpload.addEventListener('click', () => {
    fileInput.click();
  });

  if (btnClear) {
    btnClear.addEventListener('click', () => {
      fileInput.value = '';
      if (fileStatus) fileStatus.style.display = 'none';
      if (fileError) {
        fileError.textContent = '';
        fileError.style.display = 'none';
      }
    });
  }

  fileInput.addEventListener('change', async (e) => {
    const file = e.target.files && e.target.files[0];
    if (!file) return;

    if (fileError) {
      fileError.textContent = '';
      fileError.style.display = 'none';
    }

    if (file.size > MAX_FILE_SIZE_BYTES) {
      showFileError('File exceeds 10MB limit. Please upload a smaller file.');
      fileInput.value = '';
      return;
    }

    const nameLower = file.name.toLowerCase();
    const isTxt = nameLower.endsWith('.txt');
    const isPdf = nameLower.endsWith('.pdf');

    if (!isTxt && !isPdf) {
      showFileError('Unsupported file type. Only .txt and .pdf files are supported.');
      fileInput.value = '';
      return;
    }

    if (fileNameSpan) fileNameSpan.textContent = file.name;
    if (fileStatus) fileStatus.style.display = 'inline-flex';

    if (isTxt) {
      try {
        const reader = new FileReader();
        reader.onload = (event) => {
          const text = event.target.result;
          if (claimInput) {
            claimInput.value = text;
            claimInput.focus();
          }
        };
        reader.onerror = () => {
          showFileError('Failed to read text file.');
        };
        reader.readAsText(file);
      } catch (err) {
        showFileError('Failed to read text file.');
      }
    } else if (isPdf) {
      try {
        if (fileNameSpan) fileNameSpan.textContent = `Extracting ${file.name}...`;

        const base64Data = await readFileAsBase64(file);
        const { ok, response } = await fetchWithTimeout('/api/claims/extract-text', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            filename: file.name,
            content: base64Data,
          }),
        });

        if (!ok || !response.ok) {
          let errDetail = 'Failed to extract text from PDF.';
          try {
            const errJson = await response.json();
            if (errJson && errJson.detail) errDetail = errJson.detail;
          } catch (_) {}
          showFileError(errDetail);
          if (fileStatus) fileStatus.style.display = 'none';
          return;
        }

        const data = await response.json();
        if (data && typeof data.text === 'string' && data.text.trim()) {
          if (claimInput) {
            claimInput.value = data.text.trim();
            claimInput.focus();
          }
          if (fileNameSpan) fileNameSpan.textContent = file.name;
        } else {
          showFileError('No readable text could be extracted from this PDF.');
          if (fileStatus) fileStatus.style.display = 'none';
        }
      } catch (err) {
        showFileError('Error uploading and processing PDF.');
        if (fileStatus) fileStatus.style.display = 'none';
      }
    }
  });

  function showFileError(msg) {
    if (fileError) {
      fileError.textContent = msg;
      fileError.style.display = 'block';
    }
  }
}

function readFileAsBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = reader.result;
      const base64 = dataUrl.split(',')[1] || '';
      resolve(base64);
    };
    reader.onerror = (err) => reject(err);
    reader.readAsDataURL(file);
  });
}

/**
 * Executes a fetch request with a 120-second timeout via AbortController.
 * The first medicine search loads the full dataset and can take ~20-30 seconds.
 */
async function fetchWithTimeout(url, options = {}) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => {
    controller.abort();
  }, REQUEST_TIMEOUT_MS);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
    return { ok: true, response };
  } catch (err) {
    clearTimeout(timeoutId);
    return { ok: false, error: err };
  }
}

function initSearchHandlers() {
  // 1. Medicine Mode
  const btnMed = document.getElementById('btn-medicine-search');
  const inputMed = document.getElementById('input-medicine');
  const resultsMed = document.getElementById('results-medicine');

  if (btnMed && inputMed && resultsMed) {
    const handleMedSearch = async () => {
      const query = inputMed.value.trim();
      if (!query) {
        resultsMed.innerHTML = '<p class="error-text">Please enter a medicine name.</p>';
        return;
      }

      resultsMed.innerHTML = '<p class="state-text">Searching the medicine database...</p>';

      const url = `/api/medicine/search?q=${encodeURIComponent(query)}`;
      const { ok, response } = await fetchWithTimeout(url);

      if (!ok || !response.ok) {
        resultsMed.innerHTML = `<p class="error-text">${MSG_UNAVAILABLE}</p>`;
        return;
      }

      let data;
      try {
        data = await response.json();
      } catch (err) {
        resultsMed.innerHTML = `<p class="error-text">${MSG_READ_ERROR}</p>`;
        return;
      }

      if (!data || typeof data !== 'object' || !Array.isArray(data.candidates)) {
        resultsMed.innerHTML = `<p class="error-text">${MSG_READ_ERROR}</p>`;
        return;
      }

      renderMedicineResults(resultsMed, data.candidates);
    };

    btnMed.addEventListener('click', handleMedSearch);
    inputMed.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        handleMedSearch();
      }
    });
  }

  // 2. Schemes Mode
  const btnSchemes = document.getElementById('btn-schemes-search');
  const inputSchemes = document.getElementById('input-schemes');
  const resultsSchemes = document.getElementById('results-schemes');

  if (btnSchemes && inputSchemes && resultsSchemes) {
    const handleSchemeSearch = async () => {
      const query = inputSchemes.value.trim();
      if (!query) {
        resultsSchemes.innerHTML = '<p class="error-text">Please enter search terms.</p>';
        return;
      }

      resultsSchemes.innerHTML = '<p class="state-text">Checking official sources...</p>';

      const url = `/api/schemes/search?q=${encodeURIComponent(query)}`;
      const { ok, response } = await fetchWithTimeout(url);

      if (!ok || !response.ok) {
        resultsSchemes.innerHTML = `<p class="error-text">${MSG_UNAVAILABLE}</p>`;
        return;
      }

      let data;
      try {
        data = await response.json();
      } catch (err) {
        resultsSchemes.innerHTML = `<p class="error-text">${MSG_READ_ERROR}</p>`;
        return;
      }

      const hits = Array.isArray(data?.hits) ? data.hits : (Array.isArray(data?.results) ? data.results : null);
      if (!hits) {
        resultsSchemes.innerHTML = `<p class="error-text">${MSG_READ_ERROR}</p>`;
        return;
      }

      renderSchemeResults(resultsSchemes, hits);
    };

    btnSchemes.addEventListener('click', handleSchemeSearch);
    inputSchemes.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        handleSchemeSearch();
      }
    });
  }

  // 3. Claim Checker Mode
  const btnClaim = document.getElementById('btn-claim-search');
  const inputClaim = document.getElementById('input-claim');
  const resultsClaim = document.getElementById('results-claim');

  if (btnClaim && inputClaim && resultsClaim) {
    btnClaim.addEventListener('click', async () => {
      const message = inputClaim.value.trim();
      if (!message) {
        resultsClaim.innerHTML = '<p class="error-text">Please enter or paste a message to check.</p>';
        return;
      }

      resultsClaim.innerHTML = '<p class="state-text">Checking this message against retrieved evidence...</p>';

      const url = '/api/claims/check';
      const { ok, response } = await fetchWithTimeout(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message }),
      });

      if (!ok || !response.ok) {
        resultsClaim.innerHTML = `<p class="error-text">${MSG_UNAVAILABLE}</p>`;
        return;
      }

      let data;
      try {
        data = await response.json();
      } catch (err) {
        resultsClaim.innerHTML = `<p class="error-text">${MSG_READ_ERROR}</p>`;
        return;
      }

      if (!data || typeof data !== 'object' || (!('label' in data) && !('verdict' in data))) {
        resultsClaim.innerHTML = `<p class="error-text">${MSG_READ_ERROR}</p>`;
        return;
      }

      renderClaimResults(resultsClaim, data);
    });
  }
}

const VERIFIED_EXACT_MEDICINE_URLS = {
  'azithral 500 tablet': 'https://www.1mg.com/drugs/azithral-500-tablet-47672',
  'azithral 500': 'https://www.1mg.com/drugs/azithral-500-tablet-47672',
  'dolo 650 tablet': 'https://www.1mg.com/drugs/dolo-650-tablet-74467',
  'dolo 650': 'https://www.1mg.com/drugs/dolo-650-tablet-74467',
  'calpol 500mg tablet': 'https://www.1mg.com/drugs/calpol-500mg-tablet-25039',
  'calpol 500 tablet': 'https://www.1mg.com/drugs/calpol-500mg-tablet-25039',
  'calpol 500': 'https://www.1mg.com/drugs/calpol-500mg-tablet-25039',
  'aciloc 150 tablet': 'https://www.1mg.com/drugs/aciloc-150-tablet-132338',
  'aciloc 150': 'https://www.1mg.com/drugs/aciloc-150-tablet-132338',
};

function resolveMedicinePurchaseUrl(item) {
  if (item.purchase_url) return item.purchase_url;
  const brand = (item.brand || item.title || '').trim();
  const cleanBrand = brand.toLowerCase().replace(/\s+/g, ' ');
  if (VERIFIED_EXACT_MEDICINE_URLS[cleanBrand]) {
    return VERIFIED_EXACT_MEDICINE_URLS[cleanBrand];
  }
  const strength = item.strength_value ? `${item.strength_value}${item.strength_unit || ''}`.trim() : '';
  const form = (item.form || '').trim();
  const queryParts = [brand];
  if (strength && !brand.toLowerCase().includes(strength.toLowerCase())) {
    queryParts.push(strength);
  }
  if (form && !brand.toLowerCase().includes(form.toLowerCase())) {
    queryParts.push(form);
  }
  const searchQuery = queryParts.join(' ').trim();
  return `https://www.1mg.com/search/all?name=${encodeURIComponent(searchQuery)}`;
}

function renderMedicineResults(container, candidates) {
  if (candidates.length === 0) {
    container.innerHTML = '<p class="state-text">No matching medicine candidates found.</p>';
    return;
  }

  const countHtml = `<div class="results-count">Found ${candidates.length} matching candidate${candidates.length > 1 ? 's' : ''}</div>`;
  const cardsHtml = candidates.map((item, idx) => {
    const brand = escapeHtml(item.brand || item.title || 'Unknown Medicine');
    const rank = item.rank || (idx + 1);
    const salt = escapeHtml(item.salt || (item.salts && item.salts.join(', ')) || '');
    const strength = item.strength_value ? `${item.strength_value} ${item.strength_unit || ''}`.trim() : '';
    const form = escapeHtml(item.form || '');
    const manufacturer = escapeHtml(item.manufacturer || '');
    const mrp = item.mrp != null ? Number(item.mrp).toFixed(2) : null;
    const genericPrice = item.generic_price != null ? Number(item.generic_price).toFixed(2) : null;
    const purchaseUrl = item.purchase_url || resolveMedicinePurchaseUrl(item);

    return `
      <a href="${escapeHtml(purchaseUrl)}" target="_blank" rel="noopener noreferrer" class="result-card clickable-card" aria-label="Purchase ${brand} online">
        <div class="card-header">
          <h3 class="card-title">${brand}</h3>
          <div class="card-badges">
            <span class="badge badge-rank">#${rank}</span>
            <span class="badge badge-buy">Buy online &nearr;</span>
          </div>
        </div>

        <div class="card-badges" style="margin-top: 6px;">
          ${salt ? `<span class="badge badge-salt">Salt: ${salt}</span>` : ''}
          ${strength ? `<span class="badge badge-form">${escapeHtml(strength)}</span>` : ''}
          ${form ? `<span class="badge badge-form">${form}</span>` : ''}
        </div>

        ${manufacturer ? `
          <div class="meta-row">
            <span class="meta-item"><span class="meta-label">Manufacturer:</span> <span class="meta-value">${manufacturer}</span></span>
          </div>
        ` : ''}

        ${(mrp || genericPrice) ? `
          <div class="pricing-row">
            ${mrp ? `
              <div class="price-box">
                <span class="price-label">MRP</span>
                <span class="price-amount">₹${mrp}</span>
              </div>
            ` : ''}
            ${genericPrice ? `
              <div class="price-box">
                <span class="price-label">Jan Aushadhi</span>
                <span class="price-amount" style="color: #2e7d32;">₹${genericPrice}</span>
              </div>
            ` : ''}
          </div>
        ` : ''}

        <div class="card-shopping-footer">
          <span class="pharmacy-note">Opens external pharmacy &bull; Price/stock may vary</span>
          <span class="buy-link-text">Buy online &nearr;</span>
        </div>
      </a>
    `;
  }).join('');

  container.innerHTML = countHtml + `<div class="results-grid">${cardsHtml}</div>`;
}

function renderSchemeResults(container, hits) {
  if (hits.length === 0) {
    container.innerHTML = '<p class="state-text">No matching schemes found.</p>';
    return;
  }

  const countHtml = `<div class="results-count">Found ${hits.length} relevant health scheme${hits.length > 1 ? 's' : ''}</div>`;
  const cardsHtml = hits.map((item, idx) => {
    const rank = item.rank || (idx + 1);
    const title = escapeHtml(item.title || `Scheme #${rank}`);
    const zone = escapeHtml(item.zone || 'general');
    const url = item.url ? escapeHtml(item.url) : '';
    const textSnippet = escapeHtml(item.text ? (item.text.length > 280 ? item.text.slice(0, 280) + '...' : item.text) : '');

    return `
      <div class="result-card">
        <div class="card-header">
          <h3 class="card-title" style="text-transform: none;">${title}</h3>
          <div class="card-badges">
            <span class="badge badge-rank">#${rank}</span>
            <span class="badge badge-zone">${zone}</span>
          </div>
        </div>

        ${textSnippet ? `<div class="card-snippet">${textSnippet}</div>` : ''}

        <div class="card-footer">
          ${url ? `<a class="portal-link" href="${url}" target="_blank" rel="noopener noreferrer">Official Portal &rarr;</a>` : '<span></span>'}
        </div>
      </div>
    `;
  }).join('');

  container.innerHTML = countHtml + `<div class="results-grid">${cardsHtml}</div>`;
}

function renderClaimResults(container, data) {
  const label = (data.label || data.verdict || 'UNKNOWN').toUpperCase();
  const explanation = escapeHtml(data.explanation || 'No detailed explanation provided.');
  const evidenceText = data.evidence_text ? escapeHtml(data.evidence_text) : '';
  const evidenceCite = data.evidence_cite ? escapeHtml(data.evidence_cite) : '';

  let badgeClass = 'badge-warning';
  if (label.includes('VERIFIED') || label.includes('SUPPORTED') || label.includes('TRUE')) {
    badgeClass = 'badge-success';
  } else if (label.includes('REFUTED') || label.includes('CONTRADICTED') || label.includes('FALSE') || label.includes('MISLEADING')) {
    badgeClass = 'badge-danger';
  }

  container.innerHTML = `
    <div class="claim-card">
      <div class="claim-header">
        <span class="badge ${badgeClass}" style="font-size: 0.9rem; padding: 5px 12px;">${escapeHtml(label)}</span>
        <span class="claim-verdict-title">Claim Verification Result</span>
      </div>
      <div class="claim-text-block">${explanation}</div>
      ${evidenceText ? `
        <div class="evidence-box">
          <strong>Official Evidence:</strong>
          <p style="margin-top: 4px;">${evidenceText}</p>
          ${evidenceCite ? `<p style="margin-top: 4px; font-size: 0.8rem; color: var(--color-text-muted);">Source: ${evidenceCite}</p>` : ''}
        </div>
      ` : ''}
    </div>
  `;
}

function escapeHtml(text) {
  if (!text) return '';
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}
