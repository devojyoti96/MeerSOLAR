// Force default to light mode if no user preference
document.addEventListener("DOMContentLoaded", function () {
  const stored = localStorage.getItem("furo-color-mode");
  if (!stored) {
    document.documentElement.setAttribute("data-theme", "light");
    localStorage.setItem("furo-color-mode", "light");
  }

  // Remove "auto" option from toggle
  const label = document.querySelector('label[for="__theme-toggle"]');
  if (!label) return;

  const menu = label.parentElement?.querySelector("ul");
  if (!menu) return;

  const autoOption = Array.from(menu.querySelectorAll("li")).find(
    (el) => el.textContent?.toLowerCase().includes("auto")
  );
  if (autoOption) autoOption.remove();
});

