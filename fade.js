document.addEventListener("DOMContentLoaded", function () {
    // Wait for the DOM to be fully loaded before fading out the loading screen
    setTimeout(function () {
        fadeOutLoadingScreen();
    }, 500); // Adjust the delay (in milliseconds) based on your preference
});

function fadeOutLoadingScreen() {
    var loadingScreen = document.querySelector(".loading-screen");
    loadingScreen.style.opacity = 0;
    
    setTimeout(function () {
        loadingScreen.style.display = "none";
        // Fade in the main content
        var mainContent = document.querySelector(".main-body");
        mainContent.style.opacity = 1;
    }, 1200); // Adjust the duration (in milliseconds) based on your preference
}