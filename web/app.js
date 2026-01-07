/**
 * Trakt OCR Dashboard Application
 * JavaScript for the web interface
 */

const API_BASE = window.location.origin;
let refreshInterval = null;

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    // Set current year
    document.getElementById('currentYear').textContent = new Date().getFullYear();
    
    refreshStatus();
    refreshDetections();
    refreshResults();
    
    // Auto-refresh every 5 seconds
    refreshInterval = setInterval(function() {
        refreshStatus();
        refreshDetections();
    }, 5000);
});

// Fetch application status
async function refreshStatus() {
    try {
        const response = await fetch(`${API_BASE}/api/status`);
        const data = await response.json();
        
        if (data.success) {
            updateStatusDisplay(data.data);
        }
    } catch (error) {
        console.error('Error fetching status:', error);
        updateStatusIndicator('error', 'Connection Error');
    }
}

// Update status display
function updateStatusDisplay(status) {
    // Update status indicator
    const statusClass = status.status === 'running' ? 'running' : 
                        status.status === 'stopped' ? 'stopped' : '';
    updateStatusIndicator(statusClass, capitalizeFirst(status.status));
    
    // Update stats
    document.getElementById('frameCount').textContent = formatNumber(status.frame_count);
    document.getElementById('detectionCount').textContent = formatNumber(status.detection_count);
    document.getElementById('cameraStatus').textContent = status.camera_info ? 
        (status.camera_info.connected ? 'Connected' : 'Disconnected') : 'Unknown';
    document.getElementById('lastUpdate').textContent = status.last_update ? 
        formatTimestamp(status.last_update) : 'Never';
    
    // Update camera info
    if (status.camera_info) {
        updateCameraInfo(status.camera_info);
    }
}

// Update status indicator
function updateStatusIndicator(statusClass, text) {
    const dot = document.getElementById('statusDot');
    const textEl = document.getElementById('statusText');
    
    dot.className = 'status-dot ' + statusClass;
    textEl.textContent = text;
}

// Update camera information display
function updateCameraInfo(info) {
    const container = document.getElementById('cameraInfo');
    
    const items = [
        { label: 'Host', value: info.host || 'N/A' },
        { label: 'Port', value: info.port || 'N/A' },
        { label: 'Connected', value: info.connected ? 'Yes' : 'No' },
        { label: 'Streaming', value: info.streaming ? 'Yes' : 'No' },
        { label: 'Manufacturer', value: info.manufacturer || 'N/A' },
        { label: 'Model', value: info.model || 'N/A' },
        { label: 'Firmware', value: info.firmware_version || 'N/A' },
        { label: 'Serial Number', value: info.serial_number || 'N/A' }
    ];
    
    container.innerHTML = items.map(item => `
        <div class="camera-info-item">
            <label>${item.label}</label>
            <span>${item.value}</span>
        </div>
    `).join('');
}

// Fetch recent detections
async function refreshDetections() {
    try {
        const response = await fetch(`${API_BASE}/api/detections?limit=50`);
        const data = await response.json();
        
        if (data.success) {
            updateDetectionsDisplay(data.data);
        }
    } catch (error) {
        console.error('Error fetching detections:', error);
    }
}

// Update detections display
function updateDetectionsDisplay(detections) {
    const container = document.getElementById('detectionsList');
    
    if (!detections || detections.length === 0) {
        container.innerHTML = '<p class="no-data">No detections yet</p>';
        return;
    }
    
    container.innerHTML = detections.slice().reverse().map(detection => {
        const confidenceClass = getConfidenceClass(detection.confidence);
        return `
            <div class="detection-item">
                <span class="detection-text">${escapeHtml(detection.text)}</span>
                <span class="detection-confidence ${confidenceClass}">
                    ${(detection.confidence * 100).toFixed(1)}%
                </span>
            </div>
        `;
    }).join('');
}

// Fetch saved results
async function refreshResults() {
    try {
        const response = await fetch(`${API_BASE}/api/results`);
        const data = await response.json();
        
        if (data.success) {
            updateResultsDisplay(data.data);
        }
    } catch (error) {
        console.error('Error fetching results:', error);
    }
}

// Update results display
function updateResultsDisplay(results) {
    const container = document.getElementById('resultsList');
    
    if (!results || results.length === 0) {
        container.innerHTML = '<p class="no-data">No saved results</p>';
        return;
    }
    
    container.innerHTML = results.map(result => `
        <div class="result-item">
            <div class="result-header">
                <span class="result-timestamp">📅 ${formatTimestamp(result.timestamp)}</span>
                <span class="result-frame">Frame #${result.frame_number}</span>
            </div>
            <div class="result-detections">
                ${result.detections.map(d => `
                    <span class="result-detection-tag">${escapeHtml(d.text)} (${(d.confidence * 100).toFixed(0)}%)</span>
                `).join('')}
            </div>
        </div>
    `).join('');
}

// Helper functions
function formatNumber(num) {
    return num ? num.toLocaleString() : '0';
}

function formatTimestamp(timestamp) {
    if (!timestamp) return 'N/A';
    
    // Handle ISO format or custom format
    try {
        const date = new Date(timestamp.replace('_', 'T'));
        return date.toLocaleString();
    } catch {
        return timestamp;
    }
}

function capitalizeFirst(str) {
    if (!str) return '';
    return str.charAt(0).toUpperCase() + str.slice(1);
}

function getConfidenceClass(confidence) {
    if (confidence >= 0.8) return '';
    if (confidence >= 0.5) return 'medium';
    return 'low';
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
