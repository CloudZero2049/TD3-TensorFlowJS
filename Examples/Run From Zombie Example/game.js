window.ondrag = function(e) {e.preventDefault();};
window.ondblclick = function(e) {e.preventDefault();};
window.oncontextmenu = function(e) {e.preventDefault();};

const ctx = myCanvas.getContext("2d");
const ctxBG = bgCanvas.getContext("2d");

const UI = {
    enableUI: function() {
        UI.startTD3Button.disabled = false;
        UI.episodeSlider.disabled = false;
        UI.aSliderX.disabled = false;
        UI.aSliderY.disabled = false;
        UI.zSliderX.disabled = false;
        UI.zSliderY.disabled = false;
        UI.actionReplayButton.disabled = false;
    },
    startTD3Button: document.getElementById("startTD3Button"),
    actionReplayButton: document.getElementById("actionReplayButton"),
    agentWins: document.getElementById("agentWins"),
    episodesRan: document.getElementById("episodesRan"),
    pathsCheckbox: document.getElementById("pathsCheckbox"),
    episodeSlider: document.getElementById("episodeRange"),
    stepSlider: document.getElementById("stepRange"),
    randAgentCheckbox: document.getElementById("randAgentCheckbox"),
    randZomCheckbox: document.getElementById("randZomCheckbox"),
    aSliderX: document.getElementById("agentRangeX"),
    aSliderY: document.getElementById("agentRangeY"),
    zSliderX: document.getElementById("zomRangeX"),
    zSliderY: document.getElementById("zomRangeY"),
    aRangeInfoX: document.getElementById("aRangeInfoX"),
    aRangeInfoY: document.getElementById("aRangeInfoY"),
    //cRangeInfoX: document.getElementById("cRangeInfoX"),
    //cRangeInfoY: document.getElementById("cRangeInfoY"),
    zRangeInfoX: document.getElementById("zRangeInfoX"),
    zRangeInfoY: document.getElementById("zRangeInfoY"),
    modelsInput: document.getElementById("modelsInput"),
    modelsLoadedInfo: document.getElementById("modelsLoadedInfo"),
    
}

const Game = {
    version: 0.03,
    running: false, 
    user: "TD3",
    width: 500,
    height: 500,
    entities: [],
    agentWins: 0,
    entityCounter: 0,
    rewards: 0,
    punishments: 0,
    episodesRan: 0,
    maxDrawEpisodes: 25,
    agentMoves: [],
    zombieMoves: [],
    init: function() {
        this.drawBG();
        loadEnts();
      
        for (let i = this.entities.length -1; i >= 0; i--) {
            if (!this.entities[i]) {continue}
            this.entities[i].draw();
        }
        UI.startTD3Button.disabled = false;
        
    },
    start: function(user) {
        if (user) {
            this.user = user == `human` ? "human" : "TD3";
        }
        if (this.running) {return}
        this.running = true;       
        UI.startTD3Button.disabled = true;
        UI.stepSlider.style.background = "black";
        UI.episodeSlider.disabled = true;
        UI.stepSlider.disabled = true;
        UI.zSliderX.disabled = true;
        UI.zSliderY.disabled = true;
        
        if (this.user == "TD3") {
            player.user = "TD3";
            Game.agentMoves = [];
            //Game.civilianMoves = [];
            Game.zombieMoves = [];
            UI.modelsInput.disabled = true;
            //if (UI.randCivCheckbox.checked) {this.loadCivilian()}
            if (UI.randZomCheckbox.checked) {this.loadZombie()}
            if (UI.randAgentCheckbox.checked) {this.loadAgent(zom1)}
            const episodes = UI.episodeSlider.value;
            const steps = UI.stepSlider.value;

            main(episodes,steps); // <-- TD3 script

            console.info(tf.memory());
            console.info("TD3 Complete");
            this.running = false;
            UI.modelsInput.disabled = false;
            UI.enableUI();
            
        }
        
    },
    stop: function() {
        if (!this.running) {return}
        this.running = false
    },
    drawBG: function() {
        ctxBG.clearRect(0,0,Game.width,Game.height);
        ctxBG.beginPath;
        ctxBG.fillStyle = "rgb(204, 255, 221)";
        ctxBG.fillRect(0,0,bgCanvas.width,bgCanvas.height);
    },
    redrawInit: function() {
        let ents = Game.entities;
        ctx.clearRect(0,0,Game.width,Game.height);
        for (let i = ents.length -1; i >= 0; i--) {
            if (!ents[i]) {continue}
            ents[i].draw();
        }
    },
    loadAgent: function(entity) {  
        let center = [entity.x + (entity.width/2), entity.y + (entity.height/2)];
        const type = "zom";
        
        let agentLoc = this.getAgentSpawn(center, type) // Gives center
        player.x = agentLoc[0] - (player.width/2);
        player.y = agentLoc[1] - (player.height/2);
        Game.redrawInit();
    },
     loadZombie: function() {
        let zomLoc = this.getZombieSpawn()
        zom1.x = zomLoc[0] - (zom1.width/2);
        zom1.y = zomLoc[1] - (zom1.height/2);
        Game.redrawInit();
     },
    getAgentSpawn: function(center, type) { // center is [x,y] // player movement 5 * batchSize
        let maxDistance;
        let minDistance;
        if (type == "civ") {
        maxDistance = player.speed * Math.min((Game.width / player.speed),(agent.maxStepCount - 1));
        minDistance = player.speed * 5; // minimum steps to reach civilian
        }
        else {
            maxDistance = (Game.width / player.speed);
            minDistance = zom1.speed * 40; // minimum steps for zombie to reach agent
        }
        let x,y;
        do {
            const angle = Math.random() * 2 * Math.PI; // Random angle in radians
            const distance = minDistance + Math.random() * (maxDistance - minDistance); // Random place along radius
        
            // Calculate the coordinates of the random point
            x = center[0] + distance * Math.cos(angle); // these locations are center points of player/agent location
            y = center[1] + distance * Math.sin(angle);

        } while (x < (player.width + 5) || x > (Game.width -(player.width + 5)) || y < (player.height + 5) || y >= (Game.height - (player.height + 5)));
    
        return [x, y];
    },
    getZombieSpawn: function() {
        let x = zom1.width + (Math.floor(Math.random() * (Game.width - (zom1.width * 3)))); // leaving 20px gap on each side
        let y = zom1.height + (Math.floor(Math.random() * (Game.height - (zom1.height * 3))));
        x += zom1.width/2;
        y += zom1.height/2;
        return [x, y];
    }
    
} 

myCanvas.width = Game.width;
myCanvas.height = Game.height;
bgCanvas.width = Game.width;
bgCanvas.height = Game.height;

function actionReplay() {
    Game.drawBG();
    let c1 = 0;
    let startFlagged = true;
    let alpha = 0.2;
    let alphaCount = 0;
    let alphaMult = 1;                                    
    let alphaBase = Math.floor(agent.maxStepCount / 9); // 7, 14, 21, 28 , 35 , 42 , 49 , 56+ (*9, 1.0 alpha)
    let zomLoc = 0;
    let agentLoc = 0;
    Game.running = true;

   function animateReplay() {
    if (!Game.running) {console.info("Replay Complete");return}
    ctx.clearRect(0,0,Game.width,Game.height);
    let agentX = (Game.agentMoves[agentLoc][0]);
    let agentY = (Game.agentMoves[agentLoc][1]);
    let zomX = (Game.zombieMoves[zomLoc][0]);
    let zomY = (Game.zombieMoves[zomLoc][1]);
    let terminalFlag = (Game.agentMoves[agentLoc][2]);
    
    let r,g,b;
    if (startFlagged) {
        alpha = 0.2;
        alphaCount = 0;
        alphaMult = 1;
        startFlagged = false;
    }

    if (c1 > 5) {c1 = 0}

    switch(c1){
        case 0: r = 255; g = 0; b = 0 // red
        break;
        case 1: r = 255; g = 255; b = 0 // yellow (zombie will hae green trail)
        break;    
        case 2: r = 0; g = 0; b = 255 // blue
        break;  
        case 3: r = 255; g = 128; b = 0 // orange
        break; 
        case 4: r = 255; g = 0; b = 255 // pink
        break; 
        case 5: r = 0; g = 255; b = 255 // cyan
        break; 
      }
      
    // adding transparency to the dots
    if ((alphaMult < 9) && (alphaCount > alphaBase * alphaMult)) { // 1:.2, 2:.3, 3:.4, 4:.5, 5:.6, 6:.7, 7:.8, 8:.9, 9:1.0,
        alpha += 0.1;
        alphaMult++
    }  
    let rgba = `rgba(${r},${g},${b},${alpha})`;

    alphaCount++
    // Agent movement path (background canvas)
    ctxBG.beginPath();
    ctxBG.arc(agentX, agentY, 2, 0, 2 * Math.PI);
    ctxBG.fillStyle = rgba;
    ctxBG.fill();
    // Zombie movement path (background canvas)
    ctxBG.beginPath();
    ctxBG.arc(zomX, zomY, 2, 0, 2 * Math.PI);
    ctxBG.fillStyle = `rgba(0,255,0,${alpha})`;
    ctxBG.fill();

    player.x = (Game.agentMoves[agentLoc][0]) - (player.width / 2);  // getting the top left corner
    player.y = (Game.agentMoves[agentLoc][1]) - (player.height / 2); // coords are center unless refrencing object
    player.draw();

    zom1.x = (Game.zombieMoves[zomLoc][0]) - (zom1.width / 2);  // getting the top left corner
    zom1.y = (Game.zombieMoves[zomLoc][1]) - (zom1.height / 2); // coords are center unless refrencing object
    zom1.draw();
    if (terminalFlag) {
        startFlagged = true;
        c1++;
    }
    agentLoc++
    zomLoc++
    if (agentLoc > Game.agentMoves.length - 1) {Game.running = false}
    requestAnimationFrame(animateReplay);
   }
   animateReplay();
}

function animateAgent() {
    if (!UI.pathsCheckbox.checked) {Game.drawBG();}
    ctx.clearRect(0,0,Game.width,Game.height);
    let c1 = 0;
    let start = 0;
    let startFlagged = false;
    let max = agent.batch_size * 15;
    let alpha = 0.2;
    let alphaCount = 0;
    let alphaMult = 1;                                    
    let alphaBase = Math.floor(agent.maxStepCount / 9); // 7, 14, 21, 28 , 35 , 42 , 49 , 56+ (*9, 1.0 alpha)

    if (Game.agentMoves.length > max) {
        start = Game.agentMoves.length - max;
    }
     
   for (let i = start; i < Game.agentMoves.length; i++) {
    let agentX = (Game.agentMoves[i][0]);
    let agentY = (Game.agentMoves[i][1]);
    let zomX = (Game.zombieMoves[i][0]);
    let zomY = (Game.zombieMoves[i][1]);
    let terminalFlag = (Game.agentMoves[i][2]);
    
    let r,g,b;
    if (startFlagged) {
        c1++;
        alpha = 0.2;
        alphaCount = 0;
        alphaMult = 1;
        startFlagged = false;
    }
   
    if (c1 > 5) {c1 = 0}

    switch(c1){
        case 0: r = 255; g = 0; b = 0 // red
        break;
        case 1: r = 255; g = 255; b = 0 // yellow (zombie will have green trail)
        break;    
        case 2: r = 0; g = 0; b = 255 // blue
        break;  
        case 3: r = 255; g = 128; b = 0 // orange
        break; 
        case 4: r = 255; g = 0; b = 255 // pink
        break; 
        case 5: r = 0; g = 255; b = 255 // cyan
        break; 
      }
      
    // adding transparency to the dots
    if ((alphaMult < 9) && (alphaCount > alphaBase * alphaMult)) { // 1:.2, 2:.3, 3:.4, 4:.5, 5:.6, 6:.7, 7:.8, 8:.9, 9:1.0,
        alpha += 0.1;
        alphaMult++
    }
      
    let rgba = `rgba(${r},${g},${b},${alpha})`;

    alphaCount++
    // Agent movement
    ctxBG.beginPath();
    ctxBG.arc(agentX, agentY, 2, 0, 2 * Math.PI);
    ctxBG.fillStyle = rgba;
    ctxBG.fill();
    // Zombie movement
    ctxBG.beginPath();
    ctxBG.arc(zomX, zomY, 2, 0, 2 * Math.PI);
    ctxBG.fillStyle = "green";
    ctxBG.fill();

    if (terminalFlag) {
        startFlagged = true;
    }
   }

    player.x = (Game.agentMoves[Game.agentMoves.length -1][0]) - (player.width / 2);  // getting the top left corner
    player.y = (Game.agentMoves[Game.agentMoves.length -1][1]) - (player.height / 2); // coords are center unless refrencing object
    player.draw();

    zom1.x = (Game.zombieMoves[Game.zombieMoves.length - 1][0]) - (zom1.width / 2);
    zom1.y = (Game.zombieMoves[Game.zombieMoves.length - 1][1]) - (zom1.height / 2);
    zom1.draw();
    //animateCivilian(civColors);

}

function animateCivilian(civColors) {
    let start = 0;
    let max = 15;
    if (Game.civilianMoves.length > max) {
        start = Game.civilianMoves.length - max;
    }

    for (let i = start; i < Game.civilianMoves.length; i++) {
        let civMarkX = (Game.civilianMoves[i][0]);
        let civMarkY = (Game.civilianMoves[i][1]);
     
        ctxBG.save();
        ctxBG.beginPath();
        ctxBG.arc(civMarkX, civMarkY, 5, 0, 2 * Math.PI);
        ctxBG.strokeStyle = civColors[i];
        ctxBG.lineWidth = 2;
        ctxBG.stroke();
        ctxBG.restore();
    }
    const os = observationSpace;
    civ1.x = (os.civCoords[0]) - (civ1.width / 2);  // get top left corner
    civ1.y = (os.civCoords[1]) - (civ1.height / 2);
   
    civ1.draw()
}

const player = new Player(++Game.entityCounter, 150, 150, "TD3");
const zom1 = new Zombie(++Game.entityCounter, 400, 200);
function loadEnts() {
Game.entities.push(player);
Game.entities.push(zom1);
}

Game.init();