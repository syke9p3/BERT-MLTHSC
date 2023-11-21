const inputText = document.getElementById("input-text");
const analyzeBtn = document.getElementById("analyze-btn");
const clearBtn = document.getElementById("clear-btn");
const labelsContainer = document.getElementById("labels-container");
const wordCountElement = document.getElementById("word-count");
const toggleLabelsBtn = document.getElementById("toggle-labels-btn");
const hideLabelsContainer = document.getElementById("hide-labels-container");
const noLabelsContainer = document.getElementById("no-labels-container");

let allLabelsBelow50 = true;

analyzeBtn.addEventListener("click", () => {
  console.log("Input Text: " + inputText.value);
  showAnalyzingState();
  setTimeout(() => {
    fetchLabels();
  }, 3000);
});

function showAnalyzingState() {
  labelsContainer.classList.add("fade-out");
  hideLabelsContainer.style.display = "none";
  noLabelsContainer.style.display = "none";
  allLabelsBelow50 = true;
  toggleLabelsBtn.innerHTML =
    'Show Labels below 50% <i class="bx bx-chevron-down"></i>';

  setTimeout(() => {
    labelsContainer.innerHTML = `
      <div class="analyze_container">
          <p>Analyzing</p>
          <span class="loading-spinner"></span>
      </div>
  `;
    labelsContainer.classList.remove("fade-out");
  }, 500);

  // Scroll down to the labels container
  labelsContainer.scrollIntoView({
    behavior: "smooth",
    block: "start",
  });
}

toggleLabelsBtn.addEventListener("click", () => {
  const labelElements = document.querySelectorAll(".label-container");

  labelElements.forEach((label) => {
    const labelPercent = label.querySelector(".label-percent");
    const percentValue = parseFloat(labelPercent.textContent);

    if (percentValue < 50 || label.style.display === "none") {
      label.style.display = label.style.display === "none" ? "block" : "none";
    }
  });

  if (allLabelsBelow50) {
    noLabelsContainer.style.display =
      noLabelsContainer.style.display === "none" ? "block" : "none";
  }

  const buttonText = toggleLabelsBtn.innerHTML.trim();
  toggleLabelsBtn.innerHTML =
    buttonText === 'Show Labels below 50% <i class="bx bx-chevron-down"></i>'
      ? 'Hide Labels below 50% <i class="bx bx-chevron-up"></i>'
      : 'Show Labels below 50% <i class="bx bx-chevron-down"></i>';
});

function hideLabelsInitially() {
  const labelElements = document.querySelectorAll(".label-container");

  labelElements.forEach((label) => {
    const labelPercent = label.querySelector(".label-percent");
    const percentValue = parseFloat(labelPercent.textContent);

    label.style.display = percentValue < 50 ? "none" : "block";
    hideLabelsContainer.style.display = "block";

    if (percentValue >= 50) {
      allLabelsBelow50 = false;
    }
  });

  if (allLabelsBelow50) {
    labelElements.forEach((label) => {
      label.style.display = "none";
    });

    noLabelsContainer.style.display = "block";
  }
}

clearBtn.addEventListener("click", () => {
  inputText.value = "";
  document.getElementById("sample-hate-speech").selectedIndex = 0;
  resetLabels();
  updateWordCount();
  hideLabelsContainer.style.display = "none";
  noLabelsContainer.style.display = "none";
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
        const probability = parseFloat(label.probability);

        resultHTML += `
          <div class="label-container result-fade-in">
            <div class="label label-${label.name} border-none" style="width: ${probability}%;">
              <span class="label-percent label-percent-${label.name}">${label.probability}</span>&nbsp;&nbsp;${label.name}
            </div>
          </div>`;
      }

      labelsContainer.innerHTML = resultHTML;
      hideLabelsInitially();
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
  document.body.classList.toggle("dark-mode");
}

function toggleDarkMode() {
  const isDarkMode = document.body.classList.toggle("dark-mode");

  const selectedOption =
    document.getElementById("sample-hate-speech").options.selectedIndex;

  const textarea = document.getElementById("input-text");

  textarea.style.color = isDarkMode ? "white" : "black";

  /*document.getElementById("sample-hate-speech").options[selectedOption].style.backgroundColor = isDarkMode ? '#333' : 'white';*/
}
