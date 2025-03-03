document.getElementById('uploadForm').addEventListener('submit', async e => {
    e.preventDefault();
    const formData = new FormData(e.target);
    const response = await fetch('/upload', { method: 'POST', body: formData });
    const results = await response.json();
    const resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = results.map(r => `
        <div class="result-item">
            <p>${r.type}: Suitable: ${r.suitable ? 'Yes' : 'No'} (Confidence: ${r.confidence.toFixed(2)})</p>
            ${r.confidence < 0.7 ? `
                <button onclick="submitFeedback('${r.id}', true)">Correct</button>
                <button onclick="submitFeedback('${r.id}', false)">Incorrect</button>
            ` : ''}
        </div>
    `).join('');
});

document.getElementById('damageForm').addEventListener('submit', async e => {
    e.preventDefault();
    const formData = new FormData(e.target);
    const response = await fetch('/damage', { method: 'POST', body: formData });
    const result = await response.json();
    document.getElementById('damageResult').innerHTML = `
        <p>Damage Highlight:</p>
        <img src="${result.damage_url}?${Date.now()}" alt="Damage Highlight">
    `;
});

async function submitFeedback(id, correct) {
    const response = await fetch('/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id, correct })
    });
    const data = await response.json();
    alert(data.status);
}