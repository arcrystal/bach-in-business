function toggleSidebar() {
    const sidebar = document.getElementById("sidebar");
    const contentContainer = document.getElementById("content-container");
    sidebar.classList.toggle("sidebar-hidden");
    contentContainer.classList.toggle("sidebar-adjacent");
}