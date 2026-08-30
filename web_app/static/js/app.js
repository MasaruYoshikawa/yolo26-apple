// State variables
let currentFile = null;
let currentDetections = [];

// DOM Elements
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const selectFileBtn = document.getElementById('select-file-btn');
const reDetectBtn = document.getElementById('re-detect-btn');

const confSlider = document.getElementById('conf-thresh');
const confVal = document.getElementById('conf-val');
const iouSlider = document.getElementById('iou-thresh');
const iouVal = document.getElementById('iou-val');

const originalPreviewBox = document.getElementById('original-preview-box');
const imgOriginal = document.getElementById('img-original');

const detectionCount = document.getElementById('detection-count');
const imgAnnotated = document.getElementById('img-annotated');
const annotatedPlaceholder = document.getElementById('annotated-placeholder');
const annotatedContainer = document.getElementById('annotated-container');

const downloadAnnotatedBtn = document.getElementById('download-annotated-btn');
const exportCsvBtn = document.getElementById('export-csv-btn');

const historyTbody = document.getElementById('history-tbody');
const emptyHistoryRow = document.getElementById('empty-history-row');
const tableFooterSummary = document.getElementById('table-footer-summary');

// --- Slider Value Synchronization ---
confSlider.addEventListener('input', (e) => {
    confVal.textContent = parseFloat(e.target.value).toFixed(2);
});

iouSlider.addEventListener('input', (e) => {
    iouVal.textContent = parseFloat(e.target.value).toFixed(2);
});

// --- File Trigger Actions ---
selectFileBtn.addEventListener('click', (e) => {
    e.stopPropagation(); // Avoid triggering dropzone click twice
    fileInput.click();
});

dropZone.addEventListener('click', () => {
    fileInput.click();
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
    }
});

// --- Drag & Drop UI states ---
['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    }, false);
});

['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
    }, false);
});

dropZone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length > 0 && files[0].type.startsWith('image/')) {
        handleFileSelect(files[0]);
    }
});

// --- Main Image Handle ---
function handleFileSelect(file) {
    currentFile = file;
    
    // Display original image preview in Column 1
    const originalUrl = URL.createObjectURL(file);
    imgOriginal.src = originalUrl;
    originalPreviewBox.classList.remove('hidden');
    
    // Enable controls
    reDetectBtn.disabled = false;
    
    // Run initial auto detection
    runDetection();
}

// --- Run Detection HTTP Request ---
async function runDetection() {
    if (!currentFile) return;
    
    // Set loading UI states
    reDetectBtn.disabled = true;
    reDetectBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> 解析中...`;
    
    // Set placeholder text in annotated container
    annotatedPlaceholder.innerHTML = `
        <i class="fa-solid fa-spinner fa-spin icon-blue"></i>
        <p>AIが画像を検出・解析しています...<br>少々お待ちください。</p>
    `;
    annotatedPlaceholder.classList.remove('hidden');
    imgAnnotated.classList.add('hidden');
    
    // Reset output stats
    detectionCount.textContent = '...';
    
    const formData = new FormData();
    formData.append('image', currentFile);
    formData.append('conf', confSlider.value);
    formData.append('iou', iouSlider.value);
    formData.append('imgsz', 640); // default inference size
    
    try {
        const response = await fetch('/detect', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('API server returned error status ' + response.status);
        }
        
        const result = await response.json();
        
        if (result.success) {
            currentDetections = result.detections || [];
            
            // 1. Update total counts
            detectionCount.textContent = result.apple_count;
            
            // 2. Render annotated image in Column 2
            imgAnnotated.src = result.image_data;
            imgAnnotated.classList.remove('hidden');
            annotatedPlaceholder.classList.add('hidden');
            
            // 3. Set download image link in Column 3
            downloadAnnotatedBtn.href = result.image_data;
            downloadAnnotatedBtn.download = `detection_${currentFile.name}`;
            downloadAnnotatedBtn.style.pointerEvents = 'auto';
            downloadAnnotatedBtn.style.opacity = '1';
            
            // 4. Fill CSV preview table
            renderHistoryTable(currentDetections);
            
        } else {
            alert('検出に失敗しました: ' + (result.error || '不明なエラー'));
            resetOutputStates('エラーが発生しました');
        }
    } catch (err) {
        console.error(err);
        alert('接続エラー: サーバーにアクセスできません。');
        resetOutputStates('接続に失敗しました');
    } finally {
        reDetectBtn.disabled = false;
        reDetectBtn.innerHTML = `<i class="fa-solid fa-arrows-rotate"></i> 再検出`;
    }
}

// Reset states on error
function resetOutputStates(msg) {
    annotatedPlaceholder.innerHTML = `
        <i class="fa-solid fa-triangle-exclamation text-orange"></i>
        <p>${msg}</p>
    `;
    annotatedPlaceholder.classList.remove('hidden');
    imgAnnotated.classList.add('hidden');
    detectionCount.textContent = '0';
    downloadAnnotatedBtn.style.pointerEvents = 'none';
    downloadAnnotatedBtn.style.opacity = '0.5';
    renderHistoryTable([]);
}

// --- Render Coordinate Table ---
function renderHistoryTable(detections) {
    // Clear old table
    historyTbody.innerHTML = '';
    
    if (detections.length === 0) {
        historyTbody.appendChild(emptyHistoryRow);
        emptyHistoryRow.classList.remove('hidden');
        tableFooterSummary.textContent = '合計： 0件';
        exportCsvBtn.disabled = true;
        return;
    }
    
    emptyHistoryRow.classList.add('hidden');
    exportCsvBtn.disabled = false;
    
    detections.forEach((det) => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${det.no}</td>
            <td>${det.class}</td>
            <td>${det.confidence.toFixed(2)}</td>
            <td>${det.x}</td>
            <td>${det.y}</td>
            <td>${det.w}</td>
            <td>${det.h}</td>
        `;
        historyTbody.appendChild(tr);
    });
    
    tableFooterSummary.textContent = `合計： ${detections.length}件`;
}

// --- Re-detect Button Click ---
reDetectBtn.addEventListener('click', () => {
    runDetection();
});

// --- CSV Download Action ---
exportCsvBtn.addEventListener('click', () => {
    if (currentDetections.length === 0) return;
    
    // Add UTF-8 BOM prefix to support Japanese in MS Excel
    let csvContent = "data:text/csv;charset=utf-8,\uFEFF";
    csvContent += "No.,クラス,確信度,X,Y,幅,高さ\r\n";
    
    currentDetections.forEach(det => {
        const row = `${det.no},"${det.class}",${det.confidence.toFixed(2)},${det.x},${det.y},${det.w},${det.h}`;
        csvContent += row + "\r\n";
    });
    
    // Add summary row
    csvContent += `\r\n合計,,,,, ${currentDetections.length}件\r\n`;
    
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    
    const dateStr = new Date().toISOString().slice(0,10);
    const originalName = currentFile ? currentFile.name.split('.')[0] : 'image';
    link.setAttribute("download", `検出結果_${originalName}_${dateStr}.csv`);
    
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
});

// Disable download links on startup until image is processed
window.addEventListener('DOMContentLoaded', () => {
    downloadAnnotatedBtn.style.pointerEvents = 'none';
    downloadAnnotatedBtn.style.opacity = '0.5';
    exportCsvBtn.disabled = true;
});
