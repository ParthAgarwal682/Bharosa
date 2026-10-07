/**
 * Bharosa Frontend
 * Plain JavaScript interface logic.
 * Handles tab navigation, sample chip population, and plain text state management.
 * No hardcoded results, citations, URLs, or dates.
 */

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

function initSearchHandlers() {
  // Medicine Search
  const btnMed = document.getElementById('btn-medicine-search');
  const inputMed = document.getElementById('input-medicine');
  const resultsMed = document.getElementById('results-medicine');

  if (btnMed && inputMed && resultsMed) {
    const handleMedSearch = () => {
      const query = inputMed.value.trim();
      if (!query) {
        resultsMed.innerHTML = '<p class="error-text">Please enter a medicine name.</p>';
        return;
      }
      resultsMed.innerHTML = '<p class="state-text">Searching...</p>';
    };

    btnMed.addEventListener('click', handleMedSearch);
    inputMed.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        handleMedSearch();
      }
    });
  }

  // Scheme Search
  const btnSchemes = document.getElementById('btn-schemes-search');
  const inputSchemes = document.getElementById('input-schemes');
  const resultsSchemes = document.getElementById('results-schemes');

  if (btnSchemes && inputSchemes && resultsSchemes) {
    const handleSchemeSearch = () => {
      const query = inputSchemes.value.trim();
      if (!query) {
        resultsSchemes.innerHTML = '<p class="error-text">Please enter search terms.</p>';
        return;
      }
      resultsSchemes.innerHTML = '<p class="state-text">Searching...</p>';
    };

    btnSchemes.addEventListener('click', handleSchemeSearch);
    inputSchemes.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        handleSchemeSearch();
      }
    });
  }

  // Claim Checker
  const btnClaim = document.getElementById('btn-claim-search');
  const inputClaim = document.getElementById('input-claim');
  const resultsClaim = document.getElementById('results-claim');

  if (btnClaim && inputClaim && resultsClaim) {
    btnClaim.addEventListener('click', () => {
      const message = inputClaim.value.trim();
      if (!message) {
        resultsClaim.innerHTML = '<p class="error-text">Please enter a claim to check.</p>';
        return;
      }
      resultsClaim.innerHTML = '<p class="state-text">Checking...</p>';
    });
  }
}
