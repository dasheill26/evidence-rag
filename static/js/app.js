const API = ''

let selectedFile = null

const dropzone = document.getElementById('dropzone')
const fileInput = document.getElementById('fileInput')
const pasteText = document.getElementById('pasteText')
const sourceLabel = document.getElementById('sourceLabel')
const uploadBtn = document.getElementById('uploadBtn')
const uploadStatus = document.getElementById('uploadStatus')
const documentList = document.getElementById('documentList')
const docFilter = document.getElementById('docFilter')
const questionInput = document.getElementById('questionInput')
const askBtn = document.getElementById('askBtn')
const answerArea = document.getElementById('answerArea')

fileInput.addEventListener('change', () => {
  selectedFile = fileInput.files[0] || null
  if (selectedFile) dropzone.querySelector('p').textContent = selectedFile.name
})

uploadBtn.addEventListener('click', async () => {
  uploadBtn.disabled = true
  uploadStatus.textContent = 'Ingesting…'
  try {
    let res
    if (selectedFile) {
      const formData = new FormData()
      formData.append('file', selectedFile)
      res = await fetch(`${API}/api/documents`, { method: 'POST', body: formData })
    } else if (pasteText.value.trim()) {
      res = await fetch(`${API}/api/documents`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: pasteText.value, source: sourceLabel.value || 'pasted-text' }),
      })
    } else {
      uploadStatus.textContent = 'Provide a file or paste some text first.'
      uploadBtn.disabled = false
      return
    }
    const data = await res.json()
    if (data.error) throw new Error(data.error)
    uploadStatus.textContent = `Ingested "${data.source}" — ${data.chunks_created} chunks created.`
    selectedFile = null
    fileInput.value = ''
    pasteText.value = ''
    sourceLabel.value = ''
    dropzone.querySelector('p').textContent = 'Drop a .txt, .md, or .pdf file here, or click to browse'
    await refreshDocuments()
  } catch (e) {
    uploadStatus.textContent = 'Error: ' + e.message
  } finally {
    uploadBtn.disabled = false
  }
})

async function refreshDocuments() {
  const res = await fetch(`${API}/api/documents`)
  const data = await res.json()
  const docs = data.documents || []

  documentList.innerHTML = docs.length
    ? docs.map(d => `
      <div class="doc-item">
        <div><span class="doc-source">${d.source}</span> <span class="doc-meta">${d.chunks} chunks</span></div>
        <button data-doc-id="${d.doc_id}">Remove</button>
      </div>
    `).join('')
    : '<p class="placeholder">No documents ingested yet.</p>'

  documentList.querySelectorAll('button[data-doc-id]').forEach(btn => {
    btn.addEventListener('click', async () => {
      await fetch(`${API}/api/documents/${btn.dataset.docId}`, { method: 'DELETE' })
      refreshDocuments()
    })
  })

  docFilter.innerHTML = '<option value="">All documents</option>' +
    docs.map(d => `<option value="${d.doc_id}">${d.source}</option>`).join('')
}

function verdictClass(verdict) {
  if (!verdict) return 'neutral'
  if (verdict.includes('well') || verdict === 'supported') return 'good'
  if (verdict.includes('poorly') || verdict === 'unsupported') return 'bad'
  return 'neutral'
}

askBtn.addEventListener('click', async () => {
  const question = questionInput.value.trim()
  if (!question) return
  askBtn.disabled = true
  answerArea.innerHTML = '<p class="placeholder">Retrieving and generating…</p>'
  try {
    const res = await fetch(`${API}/api/ask`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, doc_id: docFilter.value || null }),
    })
    const data = await res.json()
    if (data.error) throw new Error(data.error)

    const lex = data.evaluation.lexical_faithfulness
    const judge = data.evaluation.llm_judge_faithfulness

    answerArea.innerHTML = `
      <div class="answer-block">
        <div class="answer-text">${data.answer}</div>
        <div class="answer-meta">
          <span>${data.model}</span>
          <span>${data.latency_seconds}s</span>
          <span>${data.input_tokens} in / ${data.output_tokens} out tokens</span>
          <span>£${data.estimated_cost_gbp}</span>
        </div>
        <div class="eval-grid">
          <div class="eval-card">
            <span class="eval-label">Lexical Faithfulness (always available)</span>
            <span class="eval-verdict ${verdictClass(lex?.verdict)}">${lex ? lex.score + ' — ' + lex.verdict : 'n/a'}</span>
            ${lex ? `<div class="eval-explanation">${lex.explanation}</div>` : ''}
          </div>
          <div class="eval-card">
            <span class="eval-label">LLM-Judge Faithfulness (needs live API)</span>
            <span class="eval-verdict ${verdictClass(judge?.verdict)}">${judge ? (judge.score >= 0 ? judge.score + ' — ' + judge.verdict : judge.verdict) : 'n/a'}</span>
            ${judge ? `<div class="eval-explanation">${judge.explanation}</div>` : ''}
          </div>
        </div>
        <div class="evidence-label">Retrieved evidence (${data.retrieved_chunks.length} chunks)</div>
        ${data.retrieved_chunks.map(c => `
          <div class="evidence-chunk">
            <div class="evidence-source"><span>${c.source} · chunk ${c.chunk_index}</span><span>distance ${c.distance}</span></div>
            ${c.text}
          </div>
        `).join('')}
      </div>
    `
  } catch (e) {
    answerArea.innerHTML = `<p class="placeholder">Error: ${e.message}</p>`
  } finally {
    askBtn.disabled = false
  }
})

refreshDocuments()
