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

const REQUEST_TIMEOUT_MS = 120000;
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
 * Executes a fetch request with a 120-second timeout via AbortController.
 * The first medicine search loads the full dataset and can take well over 10 seconds.
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

      resultsMed.innerHTML = '<p class="state-text">Searching...</p>';

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

  const listItems = candidates
    .map((item) => `<li>Matching search candidate: ${escapeHtml(item.brand || '')}</li>`)
    .join('');
  container.innerHTML = `<ul>${listItems}</ul>`;
}

function renderSchemeResults(container, hits) {
  if (hits.length === 0) {
    container.innerHTML = '<p class="state-text">No matching schemes found.</p>';
    return;
  }

  const listItems = hits
    .map((item) => `<li>${escapeHtml(item.title || item.doc_id || '')}</li>`)
    .join('');
  container.innerHTML = `<ul>${listItems}</ul>`;
}

function renderClaimResults(container, data) {
  const label = escapeHtml(data.label || data.verdict || '');
  const explanation = escapeHtml(data.explanation || '');
  container.innerHTML = `<p class="state-text"><strong>${label}</strong>: ${explanation}</p>`;
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}
