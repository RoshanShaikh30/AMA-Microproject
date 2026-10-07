const predictBtn = document.getElementById("predictBtn");

const imageInput = document.getElementById("imageInput");

const loading = document.getElementById("loading");

const resultCard = document.getElementById("resultCard");

const longevityValue = document.getElementById("longevityValue");

const notes = document.getElementById("notes");

predictBtn.addEventListener("click", async () => {

    const file = imageInput.files[0];

    if (!file) {

        alert("Please upload an image");

        return;
    }

    loading.style.display = "block";

    resultCard.style.display = "none";

    const formData = new FormData();

    formData.append(
        "image",
        file
    );

    try {

        const response = await fetch(

            "http://127.0.0.1:5000/predict",

            {
                method: "POST",
                body: formData
            }

        );

        const data = await response.json();

        loading.style.display = "none";

        if (data.success) {

            resultCard.style.display = "block";

            longevityValue.innerHTML =
                data.predicted_longevity + " Hours";

        
        
            function generateDots(value){

    let dots = "";

    for(let i=0;i<10;i++){

        dots += `
        <span class="${
            i < value ? "dot filled" : "dot"
        }"></span>
        `;
    }

    return dots;
}
            notes.innerHTML = `

<div class="note-row">
    <span>Citrus</span>
    <div class="dot-group">
        ${generateDots(data.detected_notes.citrus)}
    </div>
</div>

<div class="note-row">
    <span>Floral</span>
    <div class="dot-group">
        ${generateDots(data.detected_notes.floral)}
    </div>
</div>

<div class="note-row">
    <span>Woody</span>
    <div class="dot-group">
        ${generateDots(data.detected_notes.woody)}
    </div>
</div>

<div class="note-row">
    <span>Amber</span>
    <div class="dot-group">
        ${generateDots(data.detected_notes.amber)}
    </div>
</div>

<div class="note-row">
    <span>Musk</span>
    <div class="dot-group">
        ${generateDots(data.detected_notes.musk)}
    </div>
</div>

`;
        }

        else {

            alert(data.error);
        }

    }

    catch(error){

        loading.style.display = "none";

        alert("Backend connection failed");
    }

});