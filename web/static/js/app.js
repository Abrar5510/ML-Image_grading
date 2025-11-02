// ML Image Grading Web App
// Frontend JavaScript

// State management
const state = {
    currentUploadId: null,
    currentBatchId: null,
    currentView: 'single',
    previewDebounce: null
};

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    initializeEventListeners();
    loadStatistics();
});

// Event Listeners
function initializeEventListeners() {
    // Navigation
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.addEventListener('click', () => switchView(btn.dataset.view));
    });

    // Single image upload
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    const selectFileBtn = document.getElementById('select-file-btn');

    uploadArea.addEventListener('click', () => fileInput.click());
    selectFileBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });
    fileInput.addEventListener('change', handleFileSelect);

    // Drag and drop
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);

    // Viewer tabs
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => switchTab(btn.dataset.tab));
    });

    // Preview controls
    setupPreviewControls();

    // Action buttons
    document.getElementById('download-btn')?.addEventListener('click', downloadEditedImage);
    document.getElementById('new-image-btn')?.addEventListener('click', resetSingleView);

    // Batch processing
    const batchUploadArea = document.getElementById('batch-upload-area');
    const batchFileInput = document.getElementById('batch-file-input');
    const selectBatchBtn = document.getElementById('select-batch-btn');

    batchUploadArea?.addEventListener('click', () => batchFileInput.click());
    selectBatchBtn?.addEventListener('click', (e) => {
        e.stopPropagation();
        batchFileInput.click();
    });
    batchFileInput?.addEventListener('change', handleBatchFileSelect);

    document.getElementById('process-batch-btn')?.addEventListener('click', processBatch);
    document.getElementById('new-batch-btn')?.addEventListener('click', resetBatchView);
}

// View Management
function switchView(viewName) {
    // Update nav buttons
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.view === viewName);
    });

    // Update views
    document.querySelectorAll('.view').forEach(view => {
        view.classList.remove('active');
    });
    document.getElementById(`${viewName}-view`).classList.add('active');

    state.currentView = viewName;

    // Load statistics if switching to statistics view
    if (viewName === 'statistics') {
        loadStatistics();
    }
}

function switchTab(tabName) {
    // Update tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tabName);
    });

    // Update tab content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    document.getElementById(`${tabName}-tab`).classList.add('active');
}

// File Upload Handlers
function handleDragOver(e) {
    e.preventDefault();
    e.currentTarget.classList.add('drag-over');
}

function handleDragLeave(e) {
    e.currentTarget.classList.remove('drag-over');
}

function handleDrop(e) {
    e.preventDefault();
    e.currentTarget.classList.remove('drag-over');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        uploadAndProcessImage(files[0]);
    }
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        uploadAndProcessImage(file);
    }
}

// Single Image Processing
async function uploadAndProcessImage(file) {
    try {
        // Show processing section
        document.querySelector('.upload-section').classList.add('hidden');
        document.getElementById('processing-section').classList.remove('hidden');
        document.getElementById('results-section').classList.add('hidden');

        // Prepare form data
        const formData = new FormData();
        formData.append('file', file);
        formData.append('apply_edits', 'true');

        // Upload and process
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Upload failed');
        }

        const data = await response.json();
        state.currentUploadId = data.upload_id;

        // Display results
        displayResults(data);

        // Hide processing, show results
        document.getElementById('processing-section').classList.add('hidden');
        document.getElementById('results-section').classList.remove('hidden');

        showToast('Image processed successfully!', 'success');

    } catch (error) {
        console.error('Error processing image:', error);
        showToast('Error processing image: ' + error.message, 'error');
        resetSingleView();
    }
}

function displayResults(data) {
    // Display images
    document.getElementById('original-image').src = data.original_image;
    document.getElementById('comparison-image').src = data.comparison_image || data.original_image;
    document.getElementById('edited-image').src = data.edited_image || data.original_image;
    document.getElementById('preview-image').src = data.original_image;

    // Display scores
    displayScores(data.scores);

    // Display suggestions
    displaySuggestions(data.suggestions);
}

function displayScores(scores) {
    const scoreTypes = ['overall', 'composition', 'color', 'technical'];

    scoreTypes.forEach(type => {
        const score = scores[`${type}_score`] || 0;
        const fillElement = document.getElementById(`${type}-score-fill`);
        const textElement = document.getElementById(`${type}-score-text`);

        if (fillElement && textElement) {
            fillElement.style.width = `${score}%`;
            textElement.textContent = `${score.toFixed(1)}`;

            // Color based on score
            if (score >= 80) {
                fillElement.style.background = 'linear-gradient(90deg, #10b981, #059669)';
            } else if (score >= 60) {
                fillElement.style.background = 'linear-gradient(90deg, #f59e0b, #d97706)';
            } else {
                fillElement.style.background = 'linear-gradient(90deg, #ef4444, #dc2626)';
            }
        }
    });
}

function displaySuggestions(suggestions) {
    const listElement = document.getElementById('suggestions-list');
    listElement.innerHTML = '';

    if (!suggestions || suggestions.length === 0) {
        listElement.innerHTML = '<p>No suggestions available.</p>';
        return;
    }

    suggestions.forEach(suggestion => {
        const item = document.createElement('div');
        item.className = `suggestion-item ${suggestion.priority}`;

        const priorityBadge = document.createElement('span');
        priorityBadge.className = `suggestion-priority ${suggestion.priority}`;
        priorityBadge.textContent = suggestion.priority;

        const text = document.createElement('p');
        text.textContent = suggestion.suggestion;

        item.appendChild(priorityBadge);
        item.appendChild(text);
        listElement.appendChild(item);
    });
}

// Preview Controls
function setupPreviewControls() {
    const controls = [
        { id: 'brightness', valueId: 'brightness-value' },
        { id: 'contrast', valueId: 'contrast-value' },
        { id: 'saturation', valueId: 'saturation-value' },
        { id: 'temperature', valueId: 'temperature-value' },
        { id: 'sharpness', valueId: 'sharpness-value' }
    ];

    controls.forEach(control => {
        const slider = document.getElementById(`${control.id}-slider`);
        const valueDisplay = document.getElementById(control.valueId);

        slider?.addEventListener('input', (e) => {
            valueDisplay.textContent = parseFloat(e.target.value).toFixed(2);
            debouncedPreviewUpdate();
        });
    });

    document.getElementById('reset-preview-btn')?.addEventListener('click', resetPreview);
}

function debouncedPreviewUpdate() {
    clearTimeout(state.previewDebounce);
    state.previewDebounce = setTimeout(updatePreview, 300);
}

async function updatePreview() {
    if (!state.currentUploadId) return;

    try {
        const adjustments = {
            brightness: parseFloat(document.getElementById('brightness-slider').value),
            contrast: parseFloat(document.getElementById('contrast-slider').value),
            saturation: parseFloat(document.getElementById('saturation-slider').value),
            temperature: parseFloat(document.getElementById('temperature-slider').value),
            sharpness: parseFloat(document.getElementById('sharpness-slider').value)
        };

        const response = await fetch('/api/preview', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                upload_id: state.currentUploadId,
                adjustments: adjustments
            })
        });

        if (!response.ok) throw new Error('Preview failed');

        const data = await response.json();
        document.getElementById('preview-image').src = data.preview_image;

    } catch (error) {
        console.error('Error updating preview:', error);
    }
}

function resetPreview() {
    document.getElementById('brightness-slider').value = 0;
    document.getElementById('contrast-slider').value = 1.0;
    document.getElementById('saturation-slider').value = 1.0;
    document.getElementById('temperature-slider').value = 0;
    document.getElementById('sharpness-slider').value = 0;

    document.getElementById('brightness-value').textContent = '0';
    document.getElementById('contrast-value').textContent = '1.0';
    document.getElementById('saturation-value').textContent = '1.0';
    document.getElementById('temperature-value').textContent = '0';
    document.getElementById('sharpness-value').textContent = '0';

    if (state.currentUploadId) {
        const originalSrc = document.getElementById('original-image').src;
        document.getElementById('preview-image').src = originalSrc;
    }
}

// Download
async function downloadEditedImage() {
    if (!state.currentUploadId) return;

    try {
        window.location.href = `/api/download/${state.currentUploadId}`;
        showToast('Download started!', 'success');
    } catch (error) {
        console.error('Error downloading image:', error);
        showToast('Error downloading image', 'error');
    }
}

// Reset Views
function resetSingleView() {
    state.currentUploadId = null;
    document.querySelector('.upload-section').classList.remove('hidden');
    document.getElementById('processing-section').classList.add('hidden');
    document.getElementById('results-section').classList.add('hidden');
    document.getElementById('file-input').value = '';
    resetPreview();
}

// Batch Processing
function handleBatchFileSelect(e) {
    const files = Array.from(e.target.files);
    if (files.length === 0) return;

    displayBatchFiles(files);
}

function displayBatchFiles(files) {
    const fileListDiv = document.getElementById('batch-file-list');
    const filesDiv = document.getElementById('batch-files');
    const countSpan = document.getElementById('batch-file-count');

    filesDiv.innerHTML = '';
    countSpan.textContent = files.length;

    files.forEach(file => {
        const item = document.createElement('div');
        item.className = 'batch-file-item';
        item.innerHTML = `
            <span><i class="fas fa-file-image"></i> ${file.name}</span>
            <span>${(file.size / 1024 / 1024).toFixed(2)} MB</span>
        `;
        filesDiv.appendChild(item);
    });

    fileListDiv.classList.remove('hidden');
}

async function processBatch() {
    try {
        const fileInput = document.getElementById('batch-file-input');
        const files = Array.from(fileInput.files);

        if (files.length === 0) {
            showToast('No files selected', 'error');
            return;
        }

        // Show processing
        document.querySelector('.batch-upload').classList.add('hidden');
        document.getElementById('batch-processing').classList.remove('hidden');

        // Upload batch
        const formData = new FormData();
        files.forEach(file => formData.append('files', file));

        const uploadResponse = await fetch('/api/batch-upload', {
            method: 'POST',
            body: formData
        });

        if (!uploadResponse.ok) throw new Error('Batch upload failed');

        const uploadData = await uploadResponse.json();
        state.currentBatchId = uploadData.batch_id;

        // Update progress
        document.getElementById('batch-total').textContent = uploadData.files_uploaded;

        // Process batch
        const processResponse = await fetch(`/api/batch-process/${state.currentBatchId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ apply_edits: true })
        });

        if (!processResponse.ok) throw new Error('Batch processing failed');

        const processData = await processResponse.json();

        // Display results
        displayBatchResults(processData.results);

        // Hide processing, show results
        document.getElementById('batch-processing').classList.add('hidden');
        document.getElementById('batch-results').classList.remove('hidden');

        showToast(`Successfully processed ${processData.total_images} images!`, 'success');

    } catch (error) {
        console.error('Error processing batch:', error);
        showToast('Error processing batch: ' + error.message, 'error');
        resetBatchView();
    }
}

function displayBatchResults(results) {
    const gridElement = document.getElementById('batch-results-grid');
    gridElement.innerHTML = '';

    results.forEach(result => {
        if (result.status === 'success') {
            const card = document.createElement('div');
            card.className = 'batch-result-card';

            const imageUrl = `/api/batch-image/${state.currentBatchId}/${result.image_id}/edited`;

            card.innerHTML = `
                <img src="${imageUrl}" alt="${result.filename}" class="batch-result-image"
                     onerror="this.src='/api/batch-image/${state.currentBatchId}/${result.image_id}/original'">
                <div class="batch-result-info">
                    <div class="batch-result-filename">${result.filename}</div>
                    <div class="batch-result-scores">
                        <div>Overall: ${result.scores.overall_score.toFixed(1)}/100</div>
                        <div>Composition: ${result.scores.composition_score.toFixed(1)}/100</div>
                        <div>Color: ${result.scores.color_score.toFixed(1)}/100</div>
                        <div>Technical: ${result.scores.technical_score.toFixed(1)}/100</div>
                    </div>
                </div>
            `;

            gridElement.appendChild(card);
        }
    });
}

function resetBatchView() {
    state.currentBatchId = null;
    document.querySelector('.batch-upload').classList.remove('hidden');
    document.getElementById('batch-file-list').classList.add('hidden');
    document.getElementById('batch-processing').classList.add('hidden');
    document.getElementById('batch-results').classList.add('hidden');
    document.getElementById('batch-file-input').value = '';
    document.getElementById('batch-files').innerHTML = '';
}

// Statistics
async function loadStatistics() {
    try {
        const response = await fetch('/api/statistics');
        if (!response.ok) throw new Error('Failed to load statistics');

        const data = await response.json();

        document.getElementById('total-processed').textContent = data.total_feedback || 0;
        document.getElementById('total-feedback').textContent = data.total_feedback || 0;

        const avgScore = data.average_scores?.overall_score || 0;
        document.getElementById('avg-score').textContent = avgScore.toFixed(1);

        const accuracy = data.model_accuracy?.overall || 0;
        document.getElementById('model-accuracy').textContent = accuracy.toFixed(1) + '%';

    } catch (error) {
        console.error('Error loading statistics:', error);
    }
}

// Toast Notifications
function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;

    const icon = type === 'success' ? 'check-circle' :
                 type === 'error' ? 'exclamation-circle' : 'info-circle';

    toast.innerHTML = `
        <i class="fas fa-${icon}"></i>
        <span>${message}</span>
    `;

    container.appendChild(toast);

    setTimeout(() => {
        toast.style.animation = 'slideIn 0.3s ease reverse';
        setTimeout(() => toast.remove(), 300);
    }, 5000);
}

// Utility Functions
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}
