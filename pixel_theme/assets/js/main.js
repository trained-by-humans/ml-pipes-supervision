document.addEventListener("DOMContentLoaded", () => {
  const searchShell = document.querySelector(".search-shell");
  const searchToggle = document.querySelector("#search-toggle");
  const searchInput = document.querySelector("#mkdocs-search-query");

  const setSearchOpen = (open) => {
    if (!searchShell || !searchToggle) return;
    searchShell.classList.toggle("is-open", open);
    searchToggle.setAttribute("aria-expanded", String(open));
  };

  if (searchShell && searchToggle && searchInput) {
    searchToggle.addEventListener("click", () => {
      const open = !searchShell.classList.contains("is-open");
      setSearchOpen(open);
      if (open) searchInput.focus();
    });

    document.addEventListener("click", (event) => {
      if (!searchShell.contains(event.target)) setSearchOpen(false);
    });

    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        setSearchOpen(false);
        searchToggle.focus();
      }
    });
  }

  document.querySelectorAll(".main table").forEach((table) => {
    const shell = document.createElement("div");
    shell.className = "table-shell";
    table.parentNode.insertBefore(shell, table);
    const scroll = document.createElement("div");
    scroll.className = "table-scroll";
    shell.appendChild(scroll);
    scroll.appendChild(table);
  });

  document.querySelectorAll(".tabs-demo").forEach((group) => {
    const buttons = group.querySelectorAll("[data-tab]");
    const panels = group.querySelectorAll("[data-panel]");

    buttons.forEach((button) => {
      button.addEventListener("click", () => {
        buttons.forEach((item) => item.classList.toggle("active", item === button));
        panels.forEach((panel) => {
          panel.hidden = panel.dataset.panel !== button.dataset.tab;
        });
      });
    });
  });
});
