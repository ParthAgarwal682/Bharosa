/**
 * Bharosa Frontend
 * Plain JavaScript interface logic.
 * Handles tab navigation, sample chip population, request timeouts, and error handling.
 *
 * Endpoints called:
 * - Medicine: GET /api/medicine/search?q=<query>
 * - Schemes: GET /api/schemes/search?q=<query>
 * - Claim Checker: POST /api/claims/check
 */

const REQUEST_TIMEOUT_MS = 35000;
const MSG_UNAVAILABLE = 'Search is unavailable right now. Please try again later.';
const MSG_READ_ERROR = 'Something went wrong reading the results.';

document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  initSampleChips();
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
  const medChip = document.getElementById('chip-medicine');
  const medInput = document.getElementById('input-medicine');
  if (medChip && medInput) {
    medChip.addEventListener('click', () => {
      const query = medChip.getAttribute('data-query');
      if (query) {
        medInput.value = query;
        medInput.focus();
      }
    });
  }

  const schemeChip = document.getElementById('chip-schemes');
  const schemeInput = document.getElementById('input-schemes');
  if (schemeChip && schemeInput) {
    schemeChip.addEventListener('click', () => {
      const query = schemeChip.getAttribute('data-query');
      if (query) {
        schemeInput.value = query;
        schemeInput.focus();
      }
    });
  }
}

/**
 * Executes a fetch request with a strict 10-second timeout via AbortController.
 * Catches network errors, connection aborts, and timeouts.
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

      resultsMed.innerHTML = '<p class="state-text">Searching through 250,000+ medicines... Please wait.</p>';

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

      resultsSchemes.innerHTML = '<p class="state-text">Searching...</p>';

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
        resultsClaim.innerHTML = '<p class="error-text">Please enter a claim to check.</p>';
        return;
      }

      resultsClaim.innerHTML = '<p class="state-text">Checking...</p>';

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

function renderMedicineResults(container, candidates) {
  if (candidates.length === 0) {
    container.innerHTML = '<p class="state-text">No matching search candidates found.</p>';
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

    let matchBadge = '';
    if (item.similarity && typeof item.similarity.cosine === 'number') {
      const pct = Math.round(item.similarity.cosine * 100);
      matchBadge = `<span class="badge badge-score" title="Cosine Similarity: ${item.similarity.cosine.toFixed(4)}">${pct}% match</span>`;
    }

    return `
      <div class="result-card">
        <div class="card-header">
          <h3 class="card-title">${brand}</h3>
          <div class="card-badges">
            <span class="badge badge-rank">#${rank}</span>
            ${matchBadge}
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
      </div>
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
    const docIdShort = item.doc_id ? escapeHtml(item.doc_id.slice(0, 12) + '...') : '';

    let scoreBadge = '';
    const scoreVal = item.net != null ? item.net : item.cosine;
    if (typeof scoreVal === 'number') {
      scoreBadge = `<span class="badge badge-score" title="Relevance score: ${scoreVal.toFixed(4)}">Score: ${scoreVal.toFixed(2)}</span>`;
    }

    return `
      <div class="result-card">
        <div class="card-header">
          <h3 class="card-title" style="text-transform: none;">${title}</h3>
          <div class="card-badges">
            <span class="badge badge-rank">#${rank}</span>
            <span class="badge badge-zone">${zone}</span>
            ${scoreBadge}
          </div>
        </div>

        ${textSnippet ? `<div class="card-snippet">${textSnippet}</div>` : ''}

        <div class="card-footer">
          ${url ? `<a class="portal-link" href="${url}" target="_blank" rel="noopener noreferrer">Official Portal &rarr;</a>` : '<span></span>'}
          ${docIdShort ? `<span class="doc-hash" title="${escapeHtml(item.doc_id || '')}">ID: ${docIdShort}</span>` : ''}
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
  } else if (label.includes('REFUTED') || label.includes('FALSE') || label.includes('MISLEADING')) {
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
          <strong>Evidence Reference:</strong>
          <p style="margin-top: 4px;">${evidenceText}</p>
          ${evidenceCite ? `<p style="margin-top: 4px; font-size: 0.8rem; color: var(--color-text-muted);">Source: ${evidenceCite}</p>` : ''}
        </div>
      ` : ''}
    </div>
  `;
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}
