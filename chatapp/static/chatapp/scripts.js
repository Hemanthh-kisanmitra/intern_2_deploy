(function() {
	const themeToggle = document.getElementById('themeToggle');
	const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
	const savedTheme = localStorage.getItem('theme');

	function applyTheme(theme) {
		const body = document.body;
		body.classList.remove('theme-dark', 'theme-light');
		if (theme === 'dark') body.classList.add('theme-dark');
		if (theme === 'light') body.classList.add('theme-light');
	}

	applyTheme(savedTheme || (prefersDark ? 'dark' : 'light'));

	if (themeToggle) {
		themeToggle.addEventListener('click', function() {
			const isDark = document.body.classList.contains('theme-dark');
			const next = isDark ? 'light' : 'dark';
			applyTheme(next);
			localStorage.setItem('theme', next);
			const icon = themeToggle.querySelector('i');
			if (icon) {
				icon.classList.remove('fa-moon', 'fa-sun');
				icon.classList.add(next === 'dark' ? 'fa-sun' : 'fa-moon');
			}
		});
	}

	const yearEl = document.getElementById('year');
	if (yearEl) yearEl.textContent = new Date().getFullYear();
})();

