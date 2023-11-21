const inputText = document.getElementById("input-text");
const analyzeBtn = document.getElementById("analyze-btn");
const clearBtn = document.getElementById("clear-btn");
const labelsContainer = document.getElementById("labels-container");
const wordCountElement = document.getElementById("word-count");

analyzeBtn.addEventListener("click", () => {
  console.log("Input Text: " + inputText.value);
  fetchLabels();
});

clearBtn.addEventListener("click", () => {
  inputText.value = "";
  document.getElementById("sample-hate-speech").selectedIndex = 0;
  resetLabels();
  updateWordCount();
});

document.addEventListener("DOMContentLoaded", function () {
  const inputElement = document.getElementById("input-text");

  inputElement.addEventListener("input", updateWordCount);

  updateWordCount();
});

function fetchLabels() {
  fetch(`http://127.0.0.1:5000/labels?input=${inputText.value}`)
    .then((response) => response.json())
    .then((data) => {
      console.log(data);

      let labels = data.labels;
      let resultHTML = "";

      for (let label of labels) {
        const probability = parseFloat(label.probability); // Convert string to float

        resultHTML += `
                    <div class="label-container">
                        <div class="label label-${label.name} border-none" style="width: ${probability}%;">
                            <span class="label-percent label-percent-${label.name}">${label.probability}</span>&nbsp;&nbsp;${label.name}
                        </div>
                    </div>`;
      }

      labelsContainer.innerHTML = resultHTML;
    })
    .catch((error) => {
      console.error("Error:", error);
    });
}

function resetLabels() {
  labelsContainer.innerHTML = `
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
    `;
}

function updateTextArea() {
  var selectedOption = document.getElementById("sample-hate-speech");
  var textArea = document.getElementById("input-text");
  textArea.value = selectedOption.value;
  updateWordCount();
}

function updateWordCount() {
  const wordCount = inputText.value.trim().split(/\s+/).filter(Boolean).length;
  wordCountElement.textContent = wordCount;

  clearBtn.disabled = wordCount === 0;

  if (wordCount < 3 || wordCount > 280) {
    wordCountElement.style.color = "red";
    analyzeBtn.disabled = true;
  } else {
    wordCountElement.style.color = "black";
    analyzeBtn.disabled = false;
  }
}

function toggleDarkMode() {
    document.body.classList.toggle('dark-mode');
}

function toggleDarkMode() {
  const isDarkMode = document.body.classList.toggle('dark-mode');

  const selectedOption = document.getElementById("sample-hate-speech").options.selectedIndex;
  
  const textarea = document.getElementById("input-text");

  textarea.style.color = isDarkMode ? 'white' : 'black';

  /*document.getElementById("sample-hate-speech").options[selectedOption].style.backgroundColor = isDarkMode ? '#333' : 'white';*/
}

document.getElementById("toggle-dark-mode-btn").addEventListener("click", toggleDarkMode);
