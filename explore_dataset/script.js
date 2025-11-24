const path = require('path');
const fs = require('fs');

let ROOT = null;
let FILES = [];
let CURRENT = null;

function findResizedDirs(root) {
    const out = [];
    const stack = [root];

    while (stack.length > 0) {
        const dir = stack.pop();
        const items = fs.readdirSync(dir, { withFileTypes: true });

        for (const item of items) {
            if (item.isDirectory()) {
                const full = path.join(dir, item.name);
                if (item.name === "original_images") {
                    out.push(full);
                } else {
                    stack.push(full);
                }
            }
        }
    }

    return out;
}

function scanImagesOnlyInResized(root) {
    const imgDirs = findResizedDirs(root);
    let files = [];

    for (const d of imgDirs) {
        const items = fs.readdirSync(d);
        for (const it of items) {
            if (/\.(jpg|jpeg|png|webp|bmp)$/i.test(it)) {
                files.push(path.join(d, it));
            }
        }
    }

    return files;
}

const getRootPath = function (){
    let dir = path.join(__dirname, '..', 'Ophthalmic_Scans', 'raw');
    const patient = document.getElementById("patientInput").value;
    if (!patient){
        return dir;
    }

    dir =  path.join(dir, 'sub-' + patient);

    const session = document.getElementById("sessionInput").value;
    if (!session){
        return dir;
    }

    dir = path.join(dir, 'ses-' + session);

    const imageType = document.getElementById("imageTypeInput").value;
    if (!imageType){
        return dir;
    }

    dir = path.join(dir, imageType);

    const area = document.getElementById("areaInput").value;
    if (!area){
        return dir;
    }

    dir = path.join(dir, area);

    const laterality = document.getElementById("lateralityInput").value;
    if (!laterality){
        return dir;
    }

    dir = path.join(dir, laterality);
    return dir;
}


const newCurrent = function () {
    ROOT = getRootPath();

    FILES = scanImagesOnlyInResized(ROOT);

    if (!FILES.length) {
        CURRENT = null;
        return null;
    }

    CURRENT = { index: 0, filepath: FILES[0] };
    return CURRENT;
};

const nextCurrent = function () {
    if (!CURRENT) return null;

    CURRENT.index = (CURRENT.index + 1) % FILES.length;
    CURRENT.filepath = FILES[CURRENT.index];
    return CURRENT;
};

const prevCurrent = function () {
    if (!CURRENT) return null;

    CURRENT.index = (CURRENT.index - 1 + FILES.length) % FILES.length;
    CURRENT.filepath = FILES[CURRENT.index];
    return CURRENT;
};


const loadParticipantsToCombo = function(participants) {
    const datalist = document.getElementById("patients");
    datalist.innerHTML = "";
    participants.forEach(name => {
        const option = document.createElement("option");
        option.value = name;
        datalist.appendChild(option);
    });
}

const patientChanged = function (event) {
    document.getElementById("sessionInput").value = "";
    document.getElementById("imageTypeInput").value = "";
    document.getElementById("areaInput").value = "";
    document.getElementById("lateralityInput").value = "";

    if (!event.target.value) {
        return;
    }

    const datasetDir = path.join(__dirname, '..', 'Ophthalmic_Scans');
    const participant = path.join(datasetDir, 'raw', 'sub-' + event.target.value);
    if (!fs.existsSync(participant)) {
        return;
    }
    const sessions = fs.readdirSync(participant).filter(f => f.startsWith('ses-'));
    const datalist = document.getElementById("sessions");
    datalist.innerHTML = "";
    sessions.forEach(name => {
        const option = document.createElement("option");
        option.value = name.substring(4);
        datalist.appendChild(option);
    });
}

const sessionChanged = function (event) {
    document.getElementById("imageTypeInput").value = "";
    document.getElementById("areaInput").value = "";
    document.getElementById("lateralityInput").value = "";

    if (!event.target.value) {
        return;
    }

    patientIdInput = document.getElementById("patientInput").value;
    if (!patientIdInput) {
        return;
    }

    const datasetDir = path.join(__dirname, '..', 'Ophthalmic_Scans');
    const participant = path.join(datasetDir, 'raw', 'sub-' + patientIdInput, 'ses-' + event.target.value);

    if (!fs.existsSync(participant)) {
        return;
    }
    const imageTypes = fs.readdirSync(participant);
    const datalist = document.getElementById("imageTypes");
    datalist.innerHTML = "";
    imageTypes.forEach(name => {
        const option = document.createElement("option");
        option.value = name;
        datalist.appendChild(option);
    });
}

const imageTypeChanged = function (event) {
    document.getElementById("areaInput").value = "";
    document.getElementById("lateralityInput").value = "";
    if (!event.target.value) {
        return;
    }

    const datasetDir = path.join(__dirname, '..', 'Ophthalmic_Scans');
    patientIdInput = document.getElementById("patientInput").value;
    if (!patientIdInput) {
        return;
    }
    sessionInput = document.getElementById("sessionInput").value;
    if (!sessionInput) {
        return;
    }
    const participant = path.join(datasetDir, 'raw', 'sub-' + patientIdInput, 'ses-' + sessionInput, event.target.value);
    if (!fs.existsSync(participant)) {
        return;
    }
    const areas = fs.readdirSync(participant);
    const datalist = document.getElementById("areas");
    datalist.innerHTML = "";
    areas.forEach(name => {
        const option = document.createElement("option");
        option.value = name;
        datalist.appendChild(option);
    });
}

const areaChanged = function (event) {
    document.getElementById("lateralityInput").value = "";
    if (!event.target.value) {
        return;
    }

    const datasetDir = path.join(__dirname, '..', 'Ophthalmic_Scans');
    patientIdInput = document.getElementById("patientInput").value;
    if (!patientIdInput) {
        return;
    }
    sessionInput = document.getElementById("sessionInput").value;
    if (!sessionInput) {
        return;
    }

    imageTypeInput = document.getElementById("imageTypeInput").value;
    if (!imageTypeInput) {
        return;
    }
    const participant = path.join(datasetDir, 'raw', 'sub-' + patientIdInput, 'ses-' + sessionInput, imageTypeInput, event.target.value);
    if (!fs.existsSync(participant)) {
        return;
    }
    const areas = fs.readdirSync(participant);
    const datalist = document.getElementById("lateralities");
    datalist.innerHTML = "";
    areas.forEach(name => {
        const option = document.createElement("option");
        option.value = name;
        datalist.appendChild(option);
    });
}

const loadImageInfo = function (data) {
    document.getElementById("patient_id").value = data.patient_id;
    document.getElementById("image_id").value = data.image_id;
    document.getElementById("diagnosis").value = data.diagnosis;
    document.getElementById("date").value = data.date;
    document.getElementById("area").value = data.area;
    document.getElementById("reference_eye").value = data.reference_eye ? "yes" : "no";
    document.getElementById("image_type").value = data.image_type;
    document.getElementById("laterality").value = data.laterality;
}


const refreshImage = function () {
    const img = document.getElementById("image");
    img.src = CURRENT.filepath;
    const resizedImage = document.getElementById("resizedImage");
    resizedImage.src = CURRENT.filepath.replace('original_images', 'resized_images');
    const maskImage = document.getElementById("mask");
    maskImage.src = CURRENT.filepath.replace('original_images', 'masks');
    const metadataPath = CURRENT.filepath.replace('original_images', 'metadata').replace('.png', '.json')
    .replace('.jpg', '.json');
    if (fs.existsSync(metadataPath)) {
        const data = JSON.parse(fs.readFileSync(metadataPath, 'utf8'));
        loadImageInfo(data);
    }
}

const load = function () {
    newCurrent();
    refreshImage();
}

const next = function () {
    nextCurrent();
    refreshImage();
}

const prev = function () {
    prevCurrent();
    refreshImage();
}

window.onload = function () {
    const datasetDir = path.join(__dirname, '..', 'Ophthalmic_Scans');
    const participantsPath = path.join(datasetDir, 'raw', 'participants.tsv');

    try {
        const text = fs.readFileSync(participantsPath, 'utf8');
        const lines = text.trim().split('\n');
        const headers = lines[0].split('\t').map(h => h.trim());;
        const patientIdIndex = headers.indexOf('patient_id');
        window.PARTICIPANTS = lines.slice(1).map(line => {
            const cols = line.split('\t');
            return cols[patientIdIndex];
        });
        
        loadParticipantsToCombo(window.PARTICIPANTS);
        document.getElementById("patientInput").addEventListener("change", patientChanged);
        document.getElementById("sessionInput").addEventListener("change", sessionChanged);
        document.getElementById("imageTypeInput").addEventListener("change", imageTypeChanged);
        document.getElementById("areaInput").addEventListener("change", areaChanged);
        document.getElementById("loadButton").addEventListener("click", load);
        document.getElementById("prev").addEventListener("click", prev);
        document.getElementById("next").addEventListener("click", next);
    } catch (err) {
        alert('Error: ' + err);
    }
};

