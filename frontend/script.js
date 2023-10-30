const inputText = document.getElementById("input-text");
const analyzeBtn = document.getElementById("analyze-btn");
const clearBtn = document.getElementById("clear-btn");
const labelsContainer = document.getElementById("labels-container");
// let savedPosts = [] // TODO: save Posts in a database and fetch from there


analyzeBtn.addEventListener("click", () => {

    // 1. Send the input to the server (make a get request with url parameters)
    // 2. fetch then display the results

    console.log("Input Text: " + inputText.value)
    fetchLabels()

});

clearBtn.addEventListener("click", () => {
    inputText.value = ""
    labelsContainer.innerHTML =
        `
                    <div class="label-container">
                        <div class="label border-none">
                            <span class="label-percent">0.00%</span>&nbsp;&nbsp;Age
                        </div>
                    </div>
                    <div class="label-container">
                        <div class="label border-none">
                            <span class="label-percent">0.00%</span>&nbsp;&nbsp;Gender
                        </div>
                    </div>
                    <div class="label-container">
                        <div class="label border-none">
                            <span class="label-percent">0.00%</span>&nbsp;&nbsp;Physical
                        </div>
                    </div>
                    <div class="label-container">
                        <div class="label border-none">
                            <span class="label-percent">0.00%</span>&nbsp;&nbsp;Race
                        </div>
                    </div>
                    <div class="label-container">
                        <div class="label border-none">
                            <span class="label-percent">0.00%</span>&nbsp;&nbsp;Religion
                        </div>
                    </div>
                    <div class="label-container">
                        <div class="label border-none">
                            <span class="label-percent">0.00%</span>&nbsp;&nbsp;Others
                        </div>
                    </div>
        `
})


function fetchLabels() {

    fetch(`http://127.0.0.1:5000/labels?input=${inputText.value}`)
        .then(response => response.json())
        .then(data => {
            console.log(data);

            let labels = data.labels;
            let resultHTML = '';

            for (let label of labels) {
                const probability = parseFloat(label.probability); // Convert string to float

                resultHTML +=
                    `<div class="label-container">
                    <div class="label label-${label.name} border-none" style="width: ${probability}%;">
                    <span class="label-percent label-percent-${label.name}">${label.probability}</span>&nbsp;&nbsp;${label.name}
                    </div>
                </div>`;
            }

            labelsContainer.innerHTML = resultHTML;
        })
        .catch(error => {
            console.error('Error:', error);
        });
}
