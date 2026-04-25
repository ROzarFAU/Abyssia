"""
    Renna Ozar
"""
// DOM Elements
const fileInput = document.getElementById('fileInput');
const uploadArea = document.getElementById('uploadArea');
const previewWrapper = document.getElementById('previewWrapper');
const previewImg = document.getElementById('previewImage');
const analyzeBtn = document.getElementById('analyzeBtn');
const resultsArea = document.getElementById('resultsArea');
const uploadErrorDiv = document.getElementById('uploadError');
const themeToggle = document.getElementById('themeToggle');
const themeText = document.getElementById('themeText');

// Theme
let currentTheme = localStorage.getItem('ocean-theme') || 'dark';

function setTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('ocean-theme', theme);
    currentTheme = theme;
    
    if (theme === 'dark') {
        themeText.textContent = 'Dark Mode';
        themeToggle.querySelector('.theme-toggle-icon').textContent = '🌊';
    } else {
        themeText.textContent = 'Light Mode';
        themeToggle.querySelector('.theme-toggle-icon').textContent = '☀️';
    }
}

function toggleTheme() {
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    setTheme(newTheme);
}

setTheme(currentTheme);
themeToggle.addEventListener('click', toggleTheme);

let currentImageFile = null;
let currentImageDataUrl = null;

// Error UI
function showError(msg, isUpload = true) {
    if (isUpload) {
        uploadErrorDiv.textContent = msg;
        uploadErrorDiv.style.display = 'block';
        setTimeout(() => {
            uploadErrorDiv.style.display = 'none';
        }, 5000);
    } else {
        const errDiv = document.createElement('div');
        errDiv.className = 'error-message';
        errDiv.textContent = msg;
        resultsArea.innerHTML = '';
        resultsArea.appendChild(errDiv);
    }
}

function clearUploadError() {
    uploadErrorDiv.style.display = 'none';
}

// File handling
uploadArea.addEventListener('click', () => {
    fileInput.click();
});

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.style.background = 'var(--upload-hover-bg)';
    uploadArea.style.borderColor = 'var(--accent-primary)';
});

uploadArea.addEventListener('dragleave', (e) => {
    e.preventDefault();
    uploadArea.style.background = 'var(--upload-bg)';
    uploadArea.style.borderColor = 'var(--upload-border)';
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.style.background = 'var(--upload-bg)';
    uploadArea.style.borderColor = 'var(--upload-border)';
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

function handleFile(file) {
    clearUploadError();
    const validTypes = ['image/jpeg', 'image/png', 'image/jpg'];

    if (!validTypes.includes(file.type)) {
        showError('❌ Invalid file format. Please upload JPEG, JPG or PNG.', true);
        resetFileSelection();
        return;
    }

    if (file.size > 20 * 1024 * 1024) {
        showError('📁 File too large (max 20MB).', true);
        resetFileSelection();
        return;
    }

    currentImageFile = file;

    const reader = new FileReader();
    reader.onload = (ev) => {
        currentImageDataUrl = ev.target.result;
        previewImg.src = currentImageDataUrl;
        previewWrapper.style.display = 'block';
        analyzeBtn.disabled = false;
    };

    reader.readAsDataURL(file);
}

function resetFileSelection() {
    currentImageFile = null;
    currentImageDataUrl = null;
    previewWrapper.style.display = 'none';
    previewImg.src = '';
    analyzeBtn.disabled = true;
    fileInput.value = '';
}

// API call
async function sendImageForAnalysis(imageFile) {
    const formData = new FormData();
    formData.append('image', imageFile, imageFile.name);

    const response = await fetch('/analyze', {
        method: 'POST',
        body: formData,
    });

    const data = await response.json();

    if (!response.ok) {
        throw new Error(data.error || data.message || 'Server error');
    }

    return data;
}

// Render results
function renderResults(data, isUnderwater = true) {
    const { caption, top_label, labels } = data;

    // Always show the caption
    let html = `
        <div class="caption-section">
            <div class="caption-label">📝 AI Marine Caption</div>
            <div class="caption-text">${escapeHtml(caption)}</div>
        </div>
    `;

    // Only show classifications if the image is underwater
    if (isUnderwater) {
        const confidenceItems = labels.map(item => {
            let percentValue = parseFloat(item.confidence.replace('%', '')) || 0;

            let emoji = '🐟';
            if (percentValue > 80) emoji = '🐠';
            if (percentValue > 90) emoji = '🐙';

            return `
                <div class="label-item">
                    <div class="label-name">
                        <span>${emoji}</span> ${escapeHtml(item.label)}
                    </div>
                    <div style="display: flex; align-items: center; gap: 12px;">
                        <div class="confidence-bar-bg">
                            <div class="confidence-fill" style="width: ${percentValue}%;"></div>
                        </div>
                        <span class="confidence-value">${item.confidence}</span>
                    </div>
                </div>
            `;
        }).join('');

        html += `
            <div class="top-label">
                <span>🏆 Top Match</span>
                <span class="badge">${escapeHtml(top_label)}</span>
            </div>

            <div style="font-weight: 600; margin-bottom: 0.75rem;">
                📊 Classification Confidence
            </div>

            <div class="confidence-list">
                ${confidenceItems}
            </div>
        `;
    } else {
        // Show error message instead of classifications
        html += `
            <div class="error-message" style="margin: 1rem 0 0 0; text-align: center;">
                <strong>🚫 Not an Underwater Image</strong><br><br>
                This image was not detected as an underwater scene.<br>
                Please upload a valid marine or underwater scene for classification results.
            </div>
        `;
    }

    resultsArea.innerHTML = html;
}

function escapeHtml(str) {
    if (!str) return '';
    return str.replace(/[&<>]/g, m => ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;'
    }[m]));
}

function showLoadingState() {
    resultsArea.innerHTML = `
        <div style="text-align: center; padding: 2rem;">
            <div class="loader" style="margin: 0 auto 1rem;"></div>
            <p>Analyzing...</p>
        </div>
    `;
}

// Main analysis
analyzeBtn.addEventListener('click', async () => {
    if (!currentImageFile) {
        showError('No image selected.', true);
        return;
    }

    analyzeBtn.disabled = true;
    const originalBtnText = analyzeBtn.innerHTML;
    analyzeBtn.innerHTML = 'Analyzing...';

    showLoadingState();

    try {
        const result = await sendImageForAnalysis(currentImageFile);
        
        const isUnderwater = result.is_underwater !== undefined ? result.is_underwater : true;
        
        // Show caption always
        renderResults(result, isUnderwater);

    } catch (err) {
        console.error(err);

        if (err.message.toLowerCase().includes("underwater")) {
            resultsArea.innerHTML = `
                <div class="error-message" style="margin: 0; text-align: center;">
                    <strong>🚫 Not an Underwater Image</strong><br><br>
                    This image was not detected as an underwater scene.<br>
                    Please upload a valid marine or underwater scene.
                </div>
            `;
        } else {
            resultsArea.innerHTML = `
                <div class="error-message">
                    ⚠️ ${escapeHtml(err.message)}
                </div>
            `;
        }
    } finally {
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = originalBtnText;
    }
});

uploadArea.addEventListener('dragenter', clearUploadError);

function init() {
    resetFileSelection();
}
init();
