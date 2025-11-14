console.log('Notebook.js loaded');

class NotebookManager {
    constructor() {
        console.log('NotebookManager constructor called');
        this.notebookKey = 'whatsapp_analytics_notebook';
        this.currentPageId = null;
        this.pages = this.loadNotebook();
        this.autoSaveTimer = null;
        this.init();
    }

    init() {
        console.log('NotebookManager.init() called');
        this.renderNotebook();
        this.setupEventListeners();
        if (this.pages.length > 0) {
            this.openPage(this.pages[0].id);
        } else {
            this.createPage();
        }
        // Debug: Log that notebook is initialized
        console.log('Notebook initialized with', this.pages.length, 'pages');
    }

    loadNotebook() {
        try {
            const saved = localStorage.getItem(this.notebookKey);
            return saved ? JSON.parse(saved) : [];
        } catch (e) {
            console.error('Error loading notebook:', e);
            return [];
        }
    }

    saveNotebook() {
        try {
            localStorage.setItem(this.notebookKey, JSON.stringify(this.pages));
            this.updateSaveStatus('Saved ✓');
        } catch (e) {
            console.error('Error saving notebook:', e);
            this.updateSaveStatus('Error saving');
        }
    }

    updateSaveStatus(message) {
        const statusElement = document.getElementById('notebook-save-status');
        if (statusElement) {
            statusElement.textContent = message;
            if (message === 'Saved ✓') {
                setTimeout(() => {
                    statusElement.textContent = '';
                }, 2000);
            }
        }
    }

    generateId() {
        return Date.now().toString(36) + Math.random().toString(36).substr(2);
    }

    createPage(title = 'New Page') {
        const newPage = {
            id: this.generateId(),
            title: title,
            content: '',
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString(),
            pinned: false
        };
        
        this.pages.push(newPage);
        this.saveNotebook();
        this.renderNotebook();
        this.openPage(newPage.id);
        return newPage.id;
    }

    deletePage(pageId) {
        if (this.pages.length <= 1) {
            alert('You must have at least one page.');
            return;
        }
        
        if (confirm('Are you sure you want to delete this page?')) {
            this.pages = this.pages.filter(page => page.id !== pageId);
            this.saveNotebook();
            
            if (this.currentPageId === pageId) {
                this.openPage(this.pages[0].id);
            }
            
            this.renderNotebook();
        }
    }

    duplicatePage(pageId) {
        const page = this.pages.find(p => p.id === pageId);
        if (page) {
            const newPage = {
                ...page,
                id: this.generateId(),
                title: page.title + ' (Copy)',
                createdAt: new Date().toISOString(),
                updatedAt: new Date().toISOString()
            };
            
            this.pages.push(newPage);
            this.saveNotebook();
            this.renderNotebook();
            this.openPage(newPage.id);
        }
    }

    renamePage(pageId, newTitle) {
        const page = this.pages.find(p => p.id === pageId);
        if (page && newTitle.trim()) {
            page.title = newTitle.trim();
            page.updatedAt = new Date().toISOString();
            this.saveNotebook();
            this.renderNotebook();
            
            // Update the title in the editor if this is the current page
            if (this.currentPageId === pageId) {
                document.getElementById('notebook-page-title').value = page.title;
            }
        }
    }

    pinPage(pageId) {
        const page = this.pages.find(p => p.id === pageId);
        if (page) {
            page.pinned = !page.pinned;
            page.updatedAt = new Date().toISOString();
            this.saveNotebook();
            this.renderNotebook();
        }
    }

    openPage(pageId) {
        const page = this.pages.find(p => p.id === pageId);
        if (page) {
            this.currentPageId = pageId;
            
            // Update UI
            document.getElementById('notebook-page-title').value = page.title;
            document.getElementById('notebook-content').value = page.content || '';
            
            // Highlight active page in sidebar
            document.querySelectorAll('.notebook-page-item').forEach(item => {
                item.classList.remove('active');
                if (item.dataset.pageId === pageId) {
                    item.classList.add('active');
                }
            });
            
            // Focus on content editor
            document.getElementById('notebook-content').focus();
        }
    }

    updatePageContent(content) {
        const page = this.pages.find(p => p.id === this.currentPageId);
        if (page) {
            page.content = content;
            page.updatedAt = new Date().toISOString();
            
            // Auto-save with debounce
            this.updateSaveStatus('Saving...');
            clearTimeout(this.autoSaveTimer);
            this.autoSaveTimer = setTimeout(() => {
                this.saveNotebook();
            }, 1000);
        }
    }

    updatePageTitle(title) {
        const page = this.pages.find(p => p.id === this.currentPageId);
        if (page && title.trim()) {
            page.title = title.trim();
            page.updatedAt = new Date().toISOString();
            this.saveNotebook();
            this.renderNotebook();
        }
    }

    exportPage(pageId, format = 'txt') {
        const page = this.pages.find(p => p.id === pageId);
        if (page) {
            const filename = `${page.title.replace(/[^a-z0-9]/gi, '_').toLowerCase()}.${format}`;
            let content = page.content;
            
            if (format === 'md') {
                content = `# ${page.title}\n\n${page.content}`;
            }
            
            const blob = new Blob([content], { type: format === 'md' ? 'text/markdown' : 'text/plain' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }
    }

    renderNotebook() {
        const panel = document.getElementById('notebook-panel');
        console.log('renderNotebook called, panel:', panel);
        if (!panel) return;
        
        // Sort pages: pinned first, then by updated date
        const sortedPages = [...this.pages].sort((a, b) => {
            if (a.pinned && !b.pinned) return -1;
            if (!a.pinned && b.pinned) return 1;
            return new Date(b.updatedAt) - new Date(a.updatedAt);
        });
        
        panel.innerHTML = `
            <div class="notebook-editor">
                <div class="notebook-editor-header">
                    <input type="text" id="notebook-page-title" placeholder="Page Title">
                </div>
                <textarea id="notebook-content" placeholder="Start writing your notes here..."></textarea>
            </div>
            <div id="notebook-sidebar">
                <div class="notebook-header">
                    <h3>Notebook</h3>
                    <button id="notebook-new-page" class="btn btn-small" title="New Page">
                        <i class="fas fa-plus"></i>
                    </button>
                </div>
                <div class="notebook-pages-list">
                    ${sortedPages.map(page => `
                        <div class="notebook-page-item ${this.currentPageId === page.id ? 'active' : ''}" 
                             data-page-id="${page.id}">
                            <div class="page-info" onclick="notebookManager.openPage('${page.id}')">
                                ${page.pinned ? '<i class="fas fa-thumbtack pinned-icon"></i>' : ''}
                                <span class="page-title">${this.escapeHtml(page.title)}</span>
                            </div>
                            <div class="page-actions">
                                <button class="btn-icon" onclick="notebookManager.pinPage('${page.id}')" title="${page.pinned ? 'Unpin' : 'Pin'}">
                                    <i class="fas fa-thumbtack ${page.pinned ? 'pinned' : ''}"></i>
                                </button>
                                <button class="btn-icon" onclick="notebookManager.duplicatePage('${page.id}')" title="Duplicate">
                                    <i class="fas fa-copy"></i>
                                </button>
                                <button class="btn-icon" onclick="notebookManager.exportPage('${page.id}', 'txt')" title="Export as TXT">
                                    <i class="fas fa-file-export"></i>
                                </button>
                                <button class="btn-icon delete-btn" onclick="notebookManager.deletePage('${page.id}')" title="Delete">
                                    <i class="fas fa-trash"></i>
                                </button>
                            </div>
                        </div>
                    `).join('')}
                </div>
                <div class="notebook-footer">
                    <div id="notebook-save-status"></div>
                    <div class="notebook-stats">${this.pages.length} page${this.pages.length !== 1 ? 's' : ''}</div>
                </div>
            </div>
        `;
        console.log('Notebook panel rendered with', sortedPages.length, 'pages');
    }

    escapeHtml(text) {
        const map = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#039;'
        };
        return text.replace(/[&<>"']/g, m => map[m]);
    }

    setupEventListeners() {
        // Create new page button
        document.addEventListener('click', (e) => {
            if (e.target.id === 'notebook-new-page') {
                this.createPage();
            }
        });

        // Page title input
        document.addEventListener('input', (e) => {
            if (e.target.id === 'notebook-page-title') {
                this.updatePageTitle(e.target.value);
            }
        });

        // Content editor
        document.addEventListener('input', (e) => {
            if (e.target.id === 'notebook-content') {
                this.updatePageContent(e.target.value);
            }
        });

        // Rename page via double-click
        document.addEventListener('dblclick', (e) => {
            if (e.target.classList.contains('page-title')) {
                const pageId = e.target.closest('.notebook-page-item').dataset.pageId;
                const page = this.pages.find(p => p.id === pageId);
                if (page) {
                    const newTitle = prompt('Rename page:', page.title);
                    if (newTitle !== null) {
                        this.renamePage(pageId, newTitle);
                    }
                }
            }
        });
    }
}

// Initialize notebook when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Always initialize the notebook manager if the panel exists
    if (document.getElementById('notebook-panel')) {
        console.log('Notebook panel found, initializing...');
        if (typeof window.notebookManager === 'undefined') {
            window.notebookManager = new NotebookManager();
            console.log('Notebook manager created');
        } else {
            console.log('Notebook manager already exists');
        }
    } else {
        console.log('Notebook panel not found in DOM');
    }
    
    // Initialize notebook panel state
    const panel = document.getElementById('notebook-panel');
    const toggleBtn = document.getElementById('notebook-toggle');
    
    if (panel && toggleBtn) {
        const isOpen = localStorage.getItem('notebookPanelOpen') === 'true';
        if (isOpen) {
            panel.classList.add('open');
            toggleBtn.innerHTML = '<i class="fas fa-times"></i>';
            toggleBtn.title = 'Close Notebook';
        }
    }
});

// Toggle notebook panel
function toggleNotebook() {
    const panel = document.getElementById('notebook-panel');
    const toggleBtn = document.getElementById('notebook-toggle');
    
    if (panel && toggleBtn) {
        panel.classList.toggle('open');
        const isOpen = panel.classList.contains('open');
        toggleBtn.innerHTML = isOpen ? '<i class="fas fa-times"></i>' : '<i class="fas fa-book"></i>';
        toggleBtn.title = isOpen ? 'Close Notebook' : 'Open Notebook';
        
        // Save state to localStorage
        localStorage.setItem('notebookPanelOpen', isOpen);
        
        // Debug: Log toggle action
        console.log('Notebook panel toggled:', isOpen ? 'open' : 'closed');
    } else {
        console.log('Notebook panel or toggle button not found');
    }
}
