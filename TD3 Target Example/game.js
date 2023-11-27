window.ondrag = function(e) {e.preventDefault();};
window.ondblclick = function(e) {e.preventDefault();};
window.oncontextmenu = function(e) {e.preventDefault();};

const ctx = myCanvas.getContext("2d");
const ctxBG = bgCanvas.getContext("2d");

let player, civ1;
const UI = {
    //startPlayerButton: document.getElementById("startPlayerButton"),
    startTD3Button: document.getElementById("startTD3Button"),
    actionReplayButton: document.getElementById("actionReplayButton"),
    agentWins: document.getElementById("agentWins"),
    episodesRan: document.getElementById("episodesRan"),
    pathsCheckbox: document.getElementById("pathsCheckbox"),
    episodeSlider: document.getElementById("episodeRange"),
    stepSlider: document.getElementById("stepRange"),
    randAgentCheckbox: document.getElementById("randAgentCheckbox"),
    randCivCheckbox: document.getElementById("randCivCheckbox"),
    aSliderX: document.getElementById("agentRangeX"),
    aSliderY: document.getElementById("agentRangeY"),
    cSliderX: document.getElementById("civRangeX"),
    cSliderY: document.getElementById("civRangeY"),
    aRangeInfoX: document.getElementById("aRangeInfoX"),
    aRangeInfoY: document.getElementById("aRangeInfoY"),
    cRangeInfoX: document.getElementById("cRangeInfoX"),
    cRangeInfoY: document.getElementById("cRangeInfoY"),
    modelsInput: document.getElementById("modelsInput"),
    modelsLoadedInfo: document.getElementById("modelsLoadedInfo"),
    
}

const Game = {
    version: 0.02,
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
    civilianMoves: [],
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
        
        if (this.running) {return}
        this.running = true;       
       
        UI.startTD3Button.disabled = true;
        UI.episodeSlider.disabled = true;
        UI.stepSlider.disabled = true;
        UI.aSliderX.disabled = true;
        UI.aSliderY.disabled = true;
        UI.cSliderX.disabled = true;
        UI.cSliderY.disabled = true;
        
        
            player.user = "TD3";
            Game.agentMoves = [];
            Game.civilianMoves = [];
            UI.modelsInput.disabled = true;
            if (UI.randCivCheckbox.checked) {this.loadCivilian()}
            if (UI.randAgentCheckbox.checked) {this.loadAgent()}
            const episodes = UI.episodeSlider.value;
            const steps = UI.stepSlider.value;
           
            main(episodes,steps);

            console.info("TD3 Complete");
            this.running = false;
            UI.startTD3Button.disabled = false;
            UI.episodeSlider.disabled = false;
            UI.stepSlider.disabled = false;
            UI.aSliderX.disabled = false;
            UI.aSliderY.disabled = false;
            UI.cSliderX.disabled = false;
            UI.cSliderY.disabled = false;
            UI.actionReplayButton.disabled = false;
            UI.modelsInput.disabled = false;  
        
    },
    stop: function() {
        if (!this.running) {return}
        this.running = false
        
    },
    removeEnt: function(item) {
        let uid = item.uid;
       
        for (let i = this.entities.length -1; i >= 0; i--) {
            if (uid === this.entities[i].uid) {this.entities.splice(i,1);}
        }

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
    
    loadAgent: function() {  
        let center = [civ1.x + (civ1.width/2), civ1.y + (civ1.height/2)];
        let agentLoc = this.getAgentSpawn(center)
        player.x = agentLoc[0];
        player.y = agentLoc[1];
        Game.redrawInit();
    },
    loadCivilian: function() { 
         let civLoc = this.getCivilianSpawn()
         civ1.x = civLoc[0];
         civ1.y = civLoc[1];
         Game.redrawInit();
     },
    getAgentSpawn: function(center) { // center is [x,y] // player movement 5 * maxStepCount
        //const radius = 64;
        const maxDistance = player.speed * (agent.maxStepCount - 1);
        const minDistance = player.speed * 5; // minimum 5 steps to reach civilian
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
    getCivilianSpawn: function() {
        const x = civ1.width + (Math.floor(Math.random() * (Game.width - (civ1.width * 2)))); // leaving 20px gap on each side
        const y = civ1.height + (Math.floor(Math.random() * (Game.height - (civ1.height * 2))));
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
    let civLoc = 0;
    let agentLoc = 0;
    Game.running = true;

   function animateReplay() {
    if (!Game.running) {console.info("Replay Stopped");return}
    ctx.clearRect(0,0,Game.width,Game.height);
    let moveX = (Game.agentMoves[agentLoc][0]);
    let moveY = (Game.agentMoves[agentLoc][1]);
    let terminalFlag = (Game.agentMoves[agentLoc][2]);
    
    let r,g,b;
    if (startFlagged) {
        alpha = 0.2;
        alphaCount = 0;
        alphaMult = 1;
        startFlagged = false;

        if (Game.civilianMoves.length > 0){
            civ1.x = Game.civilianMoves[civLoc][0] - (civ1.width / 2);
            civ1.y = Game.civilianMoves[civLoc][1] - (civ1.height / 2);
            
        }
    }
    civ1.draw();

    if (c1 > 5) {c1 = 0}

    switch(c1){
        case 0: r = 255; g = 0; b = 0 // red
        break;
        case 1: r = 0; g = 255; b = 0 // green
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
    let rgb = `rgba(${r},${g},${b})`;

    alphaCount++
    // Agent movement path (background canvas)
    ctxBG.beginPath();
    ctxBG.arc(moveX, moveY, 2, 0, 2 * Math.PI);
    ctxBG.fillStyle = rgba;
    ctxBG.fill();

    player.x = (Game.agentMoves[agentLoc][0]) - (player.width / 2);  // getting the top left corner
    player.y = (Game.agentMoves[agentLoc][1]) - (player.height / 2); // coords are center unless refrencing object
    player.draw();

    if (terminalFlag) {
        startFlagged = true;

        if (Game.civilianMoves.length > 0){
            let civMarkX = (Game.civilianMoves[civLoc][0]);
            let civMarkY = (Game.civilianMoves[civLoc][1]);
            // Civilian x,y marker (background canvas)
            ctxBG.save();
            ctxBG.beginPath();
            ctxBG.arc(civMarkX, civMarkY, 5, 0, 2 * Math.PI);
            ctxBG.strokeStyle = rgb;
            ctxBG.lineWidth = 2;
            ctxBG.stroke();
            ctxBG.restore();
        }

        c1++;
        if ((Game.civilianMoves.length > 0) && (civLoc < Game.civilianMoves.length - 1)) {
            civLoc++
        }
    }
    agentLoc++
    if (agentLoc > Game.agentMoves.length - 1) {Game.running = false}
    requestAnimationFrame(animateReplay);
   }
   animateReplay();
}

function animateAgent() {
    //let checkbox = document.getElementById("pathsCheckbox");
    if (!UI.pathsCheckbox.checked) {Game.drawBG();}
    //ctx.clearRect(player.x-1,player.y-1,player.width+2,player.height+2);
    ctx.clearRect(0,0,Game.width,Game.height);
    let c1 = 0;
    let start = 0;
    let startFlagged = false;
    let max = agent.maxStepCount * 15;
    let alpha = 0.2;
    let alphaCount = 0;
    let alphaMult = 1;                                    
    let alphaBase = Math.floor(agent.maxStepCount / 9); // 7, 14, 21, 28 , 35 , 42 , 49 , 56+ (*9, 1.0 alpha)
    let civColors = [];

    if (Game.agentMoves.length > max) {
        start = Game.agentMoves.length - max;
    }
     
   for (let i = start; i < Game.agentMoves.length; i++) {
    let moveX = (Game.agentMoves[i][0]);
    let moveY = (Game.agentMoves[i][1]);
    let terminalFlag = (Game.agentMoves[i][2]);
   

    //console.log(`moveX: ${moveX}, moveY: ${moveY }`);
    
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
        case 1: r = 0; g = 255; b = 0 // green
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
    ctxBG.arc(moveX, moveY, 2, 0, 2 * Math.PI);
    ctxBG.fillStyle = rgba;
    ctxBG.fill();

    if (terminalFlag) {
        startFlagged = true;
        let rgb = `rgb(${r},${g},${b})`;
        civColors.push(rgb);
    }
   }

    player.x = (Game.agentMoves[Game.agentMoves.length -1][0]) - (player.width / 2);  // getting the top left corner
    player.y = (Game.agentMoves[Game.agentMoves.length -1][1]) - (player.height / 2); // coords are center unless refrencing object
    
    player.draw();
    animateCivilian(civColors);

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
// map width: 500,
// mapheight: 500,

function loadEnts() {
Game.entities.push(player = new Player(++Game.entityCounter, 150, 150, "TD3")); // init values
Game.entities.push(civ1 = new Civilian(++Game.entityCounter, 400, 200));
}

Game.init();