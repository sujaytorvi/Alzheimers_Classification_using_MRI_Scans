function previewImage() {
    const imageContainer = document.getElementById('imagePreviewContainer');
    imageContainer.innerHTML = ''; // Clear existing images

    const files = document.querySelector('input[type=file]').files;
    const maxFiles = Math.min(files.length, 10); // Limit the number of files to display

    for (let i = 0; i < maxFiles; i++) {
        const file = files[i];
        const reader = new FileReader();

        reader.onload = function (e) {
            // Create an img element and set its source to the file
            const img = document.createElement('img');
            img.src = e.target.result;
            img.alt = `Preview ${i + 1}`;
            img.style.width = '100px'; // Set a fixed width for each image or adjust as needed
            img.style.margin = '10px'; // Add some space between images
            imageContainer.appendChild(img); // Add the img element to the container
        };

        if (file) {
            reader.readAsDataURL(file);
        }
    }
}
function openTab(evt, tabName) {
    var i, tabcontent, tablinks;
    tabcontent = document.getElementsByClassName("tabcontent");
    for (i = 0; i < tabcontent.length; i++) {
        tabcontent[i].style.display = "none";
    }
    tablinks = document.getElementsByClassName("tablinks");
    for (i = 0; i < tablinks.length; i++) {
        tablinks[i].className = tablinks[i].className.replace(" active", "");
    }
    document.getElementById(tabName).style.display = "block";
    evt.currentTarget.className += " active";
}

// Automatically open the default tab on page load
document.addEventListener("DOMContentLoaded", function() {
    document.getElementById("defaultOpen").click();
});
function analyzeImage(analysisType) {
    var fileInputId = analysisType === 'alzheimer' ? 'alzImageUpload' : 'tumorImageUpload';
    var formData = new FormData();
    formData.append('file', $('#' + fileInputId)[0].files[0]);
    formData.append('analysis_type', analysisType);  // This line is crucial.

    $.ajax({
        url: '/upload',
        type: 'POST',
        data: formData,
        contentType: false,
        processData: false,
        success: function(data) {
            var resultContainerId = analysisType === 'alzheimer' ? 'alzResultContainer' : 'tumorResultContainer';
            $('#' + resultContainerId).html("<div class='result-card'>" + data.prediction + "</div>");
            // Optionally, handle image preview here
        },
        error: function() {
            alert("Error in prediction. Please try again.");
        }
    });
}

function analyzeAlzheimers() {
    analyzeImage('alzheimer');
}

function analyzeBrainTumor() {
    analyzeImage('brain_tumor');
}
