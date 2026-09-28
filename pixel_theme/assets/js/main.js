document.addEventListener("DOMContentLoaded", () => {
  const searchShell = document.querySelector(".search-shell");
  const searchToggle = document.querySelector("#search-toggle");
  const searchInput = document.querySelector("#mkdocs-search-query");
  const navToggle = document.querySelector("#nav-toggle");
  const mobileNav = document.querySelector("#mobile-nav");
  const navClose = document.querySelector("#nav-close");
  const drawerScrim = document.querySelector("#drawer-scrim");

  const setSearchOpen = (open) => {
    if (!searchShell || !searchToggle) return;
    searchShell.classList.toggle("is-open", open);
    searchToggle.setAttribute("aria-expanded", String(open));
  };

  const setNavOpen = (open) => {
    if (!navToggle || !mobileNav || !drawerScrim) return;
    mobileNav.classList.toggle("is-open", open);
    drawerScrim.classList.toggle("is-open", open);
    document.body.classList.toggle("drawer-open", open);
    mobileNav.setAttribute("aria-hidden", String(!open));
    navToggle.setAttribute("aria-expanded", String(open));
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
        if (mobileNav?.classList.contains("is-open")) {
          setNavOpen(false);
          navToggle?.focus();
        } else {
          setSearchOpen(false);
          searchToggle.focus();
        }
      }
    });
  }

  if (navToggle && mobileNav && drawerScrim) {
    navToggle.addEventListener("click", () => {
      const open = !mobileNav.classList.contains("is-open");
      setNavOpen(open);
      if (open) navClose?.focus();
    });
    navClose?.addEventListener("click", () => {
      setNavOpen(false);
      navToggle.focus();
    });
    drawerScrim.addEventListener("click", () => {
      setNavOpen(false);
      navToggle.focus();
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
