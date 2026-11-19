import { useLayoutEffect, useState } from 'react';

export function useCalendarTheme() {
  const [theme, setTheme] = useState<'dark' | 'bright'>(() => {
    const saved = localStorage.getItem('app-theme');
    if (saved === 'dark' || saved === 'bright') return saved;
    return window.matchMedia('(prefers-color-scheme: light)').matches ? 'bright' : 'dark';
  });
  useLayoutEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem('app-theme', theme);
  }, [theme]);
  return { theme, toggleTheme: () => setTheme(value => value === 'dark' ? 'bright' : 'dark') };
}
