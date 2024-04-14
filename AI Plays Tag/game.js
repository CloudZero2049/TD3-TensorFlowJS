window.ondrag = function(e) {e.preventDefault();};
window.ondblclick = function(e) {e.preventDefault();};
window.oncontextmenu = function(e) {e.preventDefault();};

const ctx = myCanvas.getContext("2d");
const ctxBG = bgCanvas.getContext("2d");

const UI = {
    tableSetup: false,
    enableUI: function() {
        UI.startButton.disabled = false;
        UI.episodeSlider.disabled = false;
        UI.stepSlider.disabled = false;
        UI.chaserSlider.disabled = false;
        UI.runnerSlider.disabled = false;
        UI.chaseSpeedSlider.disabled = false;
        UI.runSpeedSlider.disabled = false;
        UI.chaserInput.disabled = false;
        UI.runnerInput.disabled = false;

        UI.stopButton.disabled = true;
        UI.pauseButton.disabled = true;
        UI.unpauseButton.disabled = true;
        
    },
    setupPlayersTable: function() {
        const ents = Game.entities;
        for (let i = ents.length -1; i >= 0; i--) {
            const tableBody = document.getElementById("playersTable").getElementsByTagName("tbody")[0];
            const newRow = tableBody.insertRow();
            
            const cell1 = newRow.insertCell(0);
            const cell2 = newRow.insertCell(1);
            const cell3 = newRow.insertCell(2);
            
            cell1.textContent = ents[i].uid;
            cell2.textContent = ents[i].tags;
            cell3.textContent = ents[i].runTime;
        }
        if (!UI.tableSetup) {UI.tableSetup = true}
    },
    clearPlayerTable: function() {
        const tableBody = document.getElementById("playersTable").getElementsByTagName("tbody")[0];
        while (tableBody.firstChild.nextSibling) {
            tableBody.removeChild(tableBody.firstChild.nextSibling);
        }
    },
    updatePlayersTable: function(final) {
        const ents = Game.entities;
       
        const tableBody = document.getElementById("playersTable").getElementsByTagName("tbody")[0];
        const rows = tableBody.getElementsByTagName("tr");
        let bestTag = 0
        let bestRun = 0;
        let goldTags = [];
        let goldRuns = [];
        for (let i = ents.length -1; i >= 0; i--) {
            for (let j = rows.length -1; j >= 0; j--) {
                const cells = rows[j].getElementsByTagName("td");
                const rowId = cells[0].textContent;
        
                if (parseInt(rowId) == ents[i].uid) {
                    cells[0].style.backgroundColor = ents[i].color
                    cells[1].textContent = ents[i].tags;
                    cells[2].textContent = ents[i].runTime;
                    if (final) {
                        if (ents[i].tags > bestTag) {
                            bestTag = ents[i].tags
                            goldTags = [];
                            goldTags.push(ents[i].uid);
                        }
                        else if (ents[i].tags == bestTag) {
                            goldTags.push(ents[i].uid);
                        }

                        if (ents[i].runTime > bestRun) {
                            bestRun = ents[i].runTime
                            goldRuns = [];
                            goldRuns.push(ents[i].uid);
                        }
                        else if (ents[i].runTime == bestRun) {
                            goldRuns.push(ents[i].uid);
                        }
                    } // end final
                    break;
                }
            }
        } // end table update
        
        if (final) {
            for (let i = 0; i < goldTags.length; i++) {
                if (!goldTags[i]) {continue}
                for (let j = rows.length -1; j >= 0; j--) {
                    const cells = rows[j].getElementsByTagName("td");
                    const rowId = cells[0].textContent;
            
                    if (parseInt(rowId) == goldTags[i]) {
                        cells[0].style.backgroundColor = "gold";
                        cells[1].style.backgroundColor = "gold";
                        break;
                    }
                } // end tag rows
            } // end tag loop

            for (let i = 0; i < goldRuns.length; i++) {
                if (!goldRuns[i]) {continue}
                for (let j = rows.length -1; j >= 0; j--) {
                    const cells = rows[j].getElementsByTagName("td");
                    const rowId = cells[0].textContent;
            
                    if (parseInt(rowId) == goldRuns[i]) {
                        cells[0].style.backgroundColor = "gold";
                        cells[2].style.backgroundColor = "gold";
                        break;
                    }
                } // end runTime rows
            } // end runTime loop
        } // end if final
    },
    startButton: document.getElementById("startButton"),
    stopButton: document.getElementById("stopButton"),
    pauseButton: document.getElementById("pauseButton"),
    unpauseButton: document.getElementById("unpauseButton"),
    episodeSlider: document.getElementById("episodeRange"),
    stepSlider: document.getElementById("stepRange"),
    chaserSlider: document.getElementById("chaserRange"),
    runnerSlider: document.getElementById("runnerRange"),
    chaseSpeedSlider: document.getElementById("chaseSpeedRange"),
    runSpeedSlider: document.getElementById("runSpeedRange"),
    memoryLoadedInfo: document.getElementById("memoryLoadedInfo"),
    chaserInput: document.getElementById("chaserInput"),
    runnerInput: document.getElementById("runnerInput"),
    modelsLoadedInfo: document.getElementById("modelsLoadedInfo"),
}

const Game = {
    version: 0.1,
    running: false,
    paused: false,
    width: 500,
    height: 500,
    mapBorders: [],
    entities: [],
    chaserNum:1,
    runnerNum:1,
    entityCounter: 0,
    episodesRan: 0,
    init: function() {
        this.drawBG();
        UI.startButton.disabled = false; 
    },
    start: function() {
        
        if (this.running) {return}
        this.running = true;  
        UI.startButton.disabled = true;
        UI.episodeSlider.disabled = true;
        UI.stepSlider.disabled = true;
        UI.chaserSlider.disabled = true;
        UI.runnerSlider.disabled = true;
        UI.chaseSpeedSlider.disabled = true;
        UI.runSpeedSlider.disabled = true;
        UI.chaserInput.disabled = true;
        UI.runnerInput.disabled = true;
        
        UI.stopButton.disabled = false;
        UI.pauseButton.disabled = false;

        const episodes = parseInt(UI.episodeSlider.value);
        const steps = parseInt(UI.stepSlider.value);
        const chasers = parseInt(UI.chaserSlider.value);
        const runners = parseInt(UI.runnerSlider.value);
        const chaserSpeed = parseInt(UI.chaseSpeedSlider.value);
        const runnerSpeed = parseInt(UI.runSpeedSlider.value);
        Game.chaserNum = chasers;
        Game.runnerNum = runners;
        Game.drawBG();
        Game.entities = [];
        Game.entityCounter = 0;
        tagControl.chaserSpeed = chaserSpeed;
        tagControl.runnerSpeed = runnerSpeed;
        loadEnts(chasers,runners);
        this.redrawInit();

        for (let i = Game.entities.length -1; i >= 0; i--) {
            Game.entities[i].tags = 0;
            Game.entities[i].runTime = 0;
        }

        if (UI.tableSetup) {UI.clearPlayerTable();}
        UI.setupPlayersTable();

        tagMain(episodes,steps); // <-- TAG script
        
    },
    stop: function() {
        if (!this.running && !this.paused) {return}
        this.running = false;
        UI.updatePlayersTable(true);
        console.info("Tag Stopped");
        UI.enableUI();
    },
    pause: function() {
        if (!this.running) {return}
        this.running = false;
        this.paused = true;
        UI.pauseButton.disabled = true;
        UI.unpauseButton.disabled = false;
        console.info("Tag Paused");
    },
    unpause: function() {
        if (this.running || !this.paused) {return}
        this.running = true;
        this.paused = false;
        UI.pauseButton.disabled = false;
        UI.unpauseButton.disabled = true;
        tagControl.animationTime = setInterval(animatetag, tagControl.animateSpeed);
    },
    removeEnt: function(item) {
        if (!item) {console.warn("Item not found to remove"); return}
        let uid = item.uid;
        for (let i = this.entities.length -1; i >= 0; i--) {
            if (uid === this.entities[i].uid) {this.entities.splice(i,1);}
        }

    },
    drawBG: function() {
        ctxBG.clearRect(0,0,Game.width,Game.height);
        ctxBG.beginPath();
        ctxBG.fillStyle = "rgb(204, 255, 221)";
        ctxBG.fillRect(0,0,bgCanvas.width,bgCanvas.height);
    },
    redrawInit: function() {
        const ents = Game.entities;
        ctx.clearRect(0,0,Game.width,Game.height);
        for (let i = ents.length -1; i >= 0; i--) {
            if (!ents[i]) {continue}
            ents[i].draw();
            ents[i].createPolygon();
        }
        for (let i = ents.length -1; i >= 0; i--) {
            ents[i].sensor.update(Game.mapBorders, ents, ents[i].role);
        }
    },
    getColor: function(role) {
        let R,G,B;
        if (role == "chaser"){
            R = 255
            G = Math.floor(Math.random() * 170);
            B = 0;
        }
        else {
            R = 0;
            G = Math.floor(Math.random() * 255);
            B = 255
        }
        let color = `rgb(${R},${G},${B})`;

        return color
    },
    colideCheckIPIP: function(ent,moveX,moveY) {
        let xBase = ent.x + moveX;
        let yBase = ent.y + moveY;
        let x = xBase + ent.width;
        let y = yBase + ent.height;
        if (ctx.isPointInPath(xBase, yBase)) {return true}
        if (ctx.isPointInPath(x, yBase)) {return true}
        if (ctx.isPointInPath(x, y)) {return true}
        if (ctx.isPointInPath(xBase, y)) {return true}
        return false;
    },
    checkCollision: function(x,y,ents) {
        if (!ents || ents.length == 0) {return false}
        for (let i = 0; i < ents.length; i++) {
            let distance = utilsAI.distance(x, y, ents[i].location[0], ents[i].location[1]);
            if (distance <= 20) {return true}
        }
        return false
    },
    getSpawn: function(ents) {// 500-30 = 470, 10 + 0 = 10, 10+470 = 480 . 10 + 10 = 20, 480+10 = 490. max area(10,490)
        let x,y
        do {
            x = 10 + (Math.floor(Math.random() * (Game.width - 30))); // leaving 10px gap on each side
            y = 10 + (Math.floor(Math.random() * (Game.height - 30)));
            x += 20/2; // 20 is width
            y += 20/2;
        } while (this.checkCollision(x, y, ents));
        
        return [x, y]; // giving center
    }
    
} 

myCanvas.width = Game.width;
myCanvas.height = Game.height;
bgCanvas.width = Game.width;
bgCanvas.height = Game.height;
Game.mapBorders = [{x: 0, y: 0},{x: Game.width, y: 0},
    {x: Game.width, y: Game.height},{x: 0, y: Game.height}]

function loadEnts(chasers, runners) {
    const ents = Game.entities;
for (let i = 0; i < chasers; i++) {
    let spawn = Game.getSpawn(ents); // [x,y] gives center
    let x = spawn[0] - 10;
    let y = spawn[1] - 10;
    let RGB = Game.getColor("chaser");
    let color = `rgb(${RGB.R}, ${RGB.G}, ${RGB.B})`;
    ents.push(new Player(++Game.entityCounter, x, y, color,"chaser"));
}
for (let i = 0; i < runners; i++) {
    let spawn = Game.getSpawn(); // [x,y]
    let x = spawn[0] - 10;
    let y = spawn[1] - 10;
    let color = Game.getColor("runner");
    ents.push(new Player(++Game.entityCounter, x, y, color, "runner"));
}
}

Game.init();