//GAME INFO: the Twin Delayed Deep Deterministic Policy Gradient (TD3) agent will be given inputs (x,y coordinates of self, zombies, and civilians), and will have outputs (moving across x,y plane)
// Agent will get rewards for touching civilians, and punishment for touching zombies.

window.ondrag = function(e) {e.preventDefault();};
window.ondblclick = function(e) {e.preventDefault();};
window.oncontextmenu = function(e) {e.preventDefault();};

const ctx = myCanvas.getContext("2d");
const ctxBG = bgCanvas.getContext("2d");

const UI = {
    enableUI: function() {
        UI.startTD3Button.disabled = false;
        UI.recordUserButton.disabled = false;
        UI.episodeSlider.disabled = false;
        UI.stepSlider.disabled = false;
        UI.aSliderX.disabled = false;
        UI.aSliderY.disabled = false;
        //UI.cSliderX.disabled = false;
        //UI.cSliderY.disabled = false;
        UI.zSliderX.disabled = false;
        UI.zSliderY.disabled = false;
        UI.downloadMemoryButton.disabled = false;
        UI.downloadActorButton.disabled = false;
        UI.downloadModelsButton.disabled = false;
        UI.actionReplayButton.disabled = false;
        UI.testAgentButton.disabled = false;
        UI.logLossButton.disabled = false;
        //UI.testMarathonButton.disabled = false;
    },
    startTD3Button: document.getElementById("startTD3Button"),
    actionReplayButton: document.getElementById("actionReplayButton"),
    recordUserButton: document.getElementById("recordUserButton"),
    trainFromMemoryButton: document.getElementById("trainFromMemoryButton"),
    testAgentButton: document.getElementById("testAgentButton"),
    logMemoryButton: document.getElementById("logMemoryButton"),
    logLossButton: document.getElementById("logLossButton"),
    //testMarathonButton: document.getElementById("testMarathonButton"),
    //marathonButton: document.getElementById("marathonButton"),
    agentWins: document.getElementById("agentWins"),
    episodesRan: document.getElementById("episodesRan"),
    winsThisRun: document.getElementById("winsThisRun"),
    trainModeCheckbox: document.getElementById("trainModeCheckbox"),
    pathsCheckbox: document.getElementById("pathsCheckbox"),
    episodeSlider: document.getElementById("episodeRange"),
    stepSlider: document.getElementById("stepRange"),
    batchSlider: document.getElementById("batchRange"),
    warmupSlider: document.getElementById("warmupRange"),
    gammaSlider: document.getElementById("gammaRange"),
    alphaInfo: document.getElementById("alphaInfo"),
    betaInfo: document.getElementById("betaInfo"),
    randAgentCheckbox: document.getElementById("randAgentCheckbox"),
    //randCivCheckbox: document.getElementById("randCivCheckbox"),
    randZomCheckbox: document.getElementById("randZomCheckbox"),
    aSliderX: document.getElementById("agentRangeX"),
    aSliderY: document.getElementById("agentRangeY"),
    //cSliderX: document.getElementById("civRangeX"),
    //cSliderY: document.getElementById("civRangeY"),
    zSliderX: document.getElementById("zomRangeX"),
    zSliderY: document.getElementById("zomRangeY"),
    aRangeInfoX: document.getElementById("aRangeInfoX"),
    aRangeInfoY: document.getElementById("aRangeInfoY"),
    //cRangeInfoX: document.getElementById("cRangeInfoX"),
    //cRangeInfoY: document.getElementById("cRangeInfoY"),
    zRangeInfoX: document.getElementById("zRangeInfoX"),
    zRangeInfoY: document.getElementById("zRangeInfoY"),
    downloadMemoryButton: document.getElementById("downloadMemoryButton"),
    memoryLoadedInfo: document.getElementById("memoryLoadedInfo"),
    downloadActorButton: document.getElementById("downloadActorButton"),
    downloadModelsButton: document.getElementById("downloadModelsButton"),
    modelsInput: document.getElementById("modelsInput"),
    modelsLoadedInfo: document.getElementById("modelsLoadedInfo"),
    
}

const Game = {
    version: 0.09,
    running: false, 
    user: "TD3",
    testing: false,
    marathon: false,
    width: 500,
    height: 500,
    mapBorders: [],
    entities: [],
    numCivs: 25,
    agentWins: 0,
    winsThisRun: 0,
    entityCounter: 0,
    rewards: 0,
    episodesRan: 0,
    maxDrawEpisodes: 25,
    agentMoves: [],
    civilianMoves: [],
    zombieMoves: [],
    init: function() {
        this.drawBG();
        loadEnts();
        
        this.redrawInit();
       
        //UI.startPlayerButton.disabled = false;
        UI.startTD3Button.disabled = false;
        UI.recordUserButton.disabled = false;
        
    },
    start: function(user,testing = false,marathon = false) {
        
        if (this.running) {return}
        this.running = true;     
        if (user) {
            this.user = user == `human` ? "human" : "TD3";
        }  
        if (testing) {this.testing = true;console.log("Testing Active...")}
        //if (marathon) {this.marathon = true;console.log("Marathon Active...")}

        UI.startTD3Button.disabled = true;
        UI.recordUserButton.disabled = true;
        
        UI.batchSlider.style.background = "black";
        UI.warmupSlider.style.background = "black";
        UI.episodeSlider.disabled = true;
        UI.stepSlider.disabled = true;
        UI.batchSlider.disabled = true;
        UI.warmupSlider.disabled = true;
        
        //UI.cSliderX.disabled = true;
        //UI.cSliderY.disabled = true;
        UI.zSliderX.disabled = true;
        UI.zSliderY.disabled = true;
        
        if (this.user == "TD3") {
            player.user = "TD3";
            Game.agentMoves = [];
            //Game.civilianMoves = [];
            Game.zombieMoves = [];
            Game.winsThisRun = 0;
            UI.winsThisRun.innerHTML = `Wins This Run: ${Game.winsThisRun}`;
            UI.modelsInput.disabled = true;
            
            //if (UI.randZomCheckbox.checked) {this.loadZombie()}
            
            const episodes = parseInt(UI.episodeSlider.value);
            const steps = parseInt(UI.stepSlider.value);
            const batchSize = parseInt(UI.batchSlider.value);
            const warmupSteps = parseInt(UI.warmupSlider.value);
            const gamma = parseFloat(UI.gammaSlider.value);
            /*
            if (Game.marathon && Game.entities.length < Game.numCivs + 2) {
                for (let i = 0; i < Game.numCivs; i++){
                    let civLoc = Game.getCivilianSpawn();
                    let x = civLoc[0] - 10;
                    let y = civLoc[1] - 10;
                    Game.entities.push(new Civilian(++Game.entityCounter, x, y));
              
                  }
                  Game.redrawInit();
            }

            if (Game.testing && Game.marathon) {
                Game.drawBG();
                marathonMain(episodes,steps);   // <-- marathonMode.js using TD3
            }
            */
            //else {
                //if (UI.randCivCheckbox.checked) {this.loadCivilian()}
                if (UI.randZomCheckbox.checked) {this.loadZombie()}
                if (UI.randAgentCheckbox.checked) {this.loadAgent(zom1)}
                main(episodes,steps,batchSize,warmupSteps,gamma); // <-- TD3 script
                console.info(tf.memory());
                if (!this.testing) { console.info(`TD3 Complete. Gamma: ${agent.gamma}`);}
                else {console.info(`Testing Complete. Wins: ${Game.winsThisRun}`);}

                this.running = false;
                this.testing = false;
                //this.marathon = false;
                UI.trainFromMemoryButton.disabled = false;
                UI.enableUI();
            //}
            
            
        }
        else if(this.user == "human") {
            player.user = "human";
            
            UI.modelsInput.disabled = true;
            //if (UI.randCivCheckbox.checked) {this.loadCivilian()}
            if (UI.randZomCheckbox.checked) {this.loadZombie()}
            if (UI.randAgentCheckbox.checked) {this.loadAgent(zom1)}
            const episodes = parseInt(UI.episodeSlider.value);
            const steps = parseInt(UI.stepSlider.value);
            const batchSize = parseInt(UI.batchSlider.value);

            userTrainMain(episodes,steps,batchSize); 

           
        }
        
    },
    stop: function() {
        if (!this.running) {return}
        this.running = false
        // clearInterval(animateTime);
        //console.info("Animate Stopped");
    },
    removeEnt: function(item) {
        let uid = item.uid;
        //let type = item.arrayType;
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
    updateRewardtext: function() {
        //rewardTextArea.innerHTML = `Rewards: ${this.rewards}`;
       
    },
    redrawInit: function() {
        let ents = Game.entities;
        ctx.clearRect(0,0,Game.width,Game.height);
        for (let i = ents.length -1; i >= 0; i--) {
            if (!ents[i]) {continue}
            ents[i].draw();
            if (ents[i].type == "player"){continue}
            ents[i].createPolygon();
        }
        //zom1.createPolygon();
        //civ1.createPolygon();
        player.sensor.update(Game.mapBorders,ents);

        if(!Game.running){player.sensor.draw(ctx);}
        
        /*
        const offsets = player.sensor.readings.map((reading) => {
            if (reading == null) {
                return 1;  // Default value for no detection
            } else {
                // Adjust the offset based on the type of the detected object
                if (reading.type === 'civilian') {
                    // Map distances from 0 to 1 (ascending)
                    return parseFloat(reading.offset.toFixed(3));
                } else if (reading.type === 'wall') {
                    // Map distances from 0 to -1 (descending)
                    //return parseFloat((-(1 - reading.offset)).toFixed(3));
                } else {
                    // Handle other types if necessary
                    return parseFloat(1 - reading.offset.toFixed(3));  // Default value for unknown types
                }
            }
        });
        console.log(offsets);
        */
       /*
        function getDistance(x1,y1,x2,y2) {
            return Math.floor(Math.hypot(x2 - x1, y2 - y1));
          }
        console.log(parseFloat(((getDistance(UI.aSliderX.value, UI.aSliderY.value, UI.zSliderX.value, UI.zSliderY.value) * 0.002)).toFixed(8)));
        */
    },
    
    loadPlayer: function() {
       // Game.entities.push(player = new Player(++Game.entityCounter,150,150,"human"));
    },
    loadAgent: function(entity) {  
        let center = [entity.x + (entity.width/2), entity.y + (entity.height/2)];
        const type = entity.type;
        
        let agentLoc = this.getAgentSpawn(center, type) // Gives center
        player.x = agentLoc[0] - (player.width/2);
        player.y = agentLoc[1] - (player.height/2);
        Game.redrawInit();
    },
    loadCivilian: function() { 
         let civLoc = this.getCivilianSpawn() 
         observationSpace.civCoords[0] = civLoc[0];
         observationSpace.civCoords[1] = civLoc[1];

         civ1.x = civLoc[0] - (civ1.width/2);
         civ1.y = civLoc[1] - (civ1.height/2);
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
        if (type == "civilian") {
        //maxDistance = player.speed * Math.min((Game.width / player.speed),(agent.maxStepCount - 1));
        maxDistance = player.sensor.rayLength - 20;
        minDistance = player.speed * 5; // minimum steps to reach civilian
        }
        else if (type == "zombie"){ // "zombie"
            maxDistance = Game.width / 1.5;
            minDistance = zom1.speed * 45; // minimum steps for zombie to reach agent
        }
        let x,y;
        do {
            const angle = Math.random() * 2 * Math.PI; // Random angle in radians
            const distance = minDistance + Math.random() * (maxDistance - minDistance); // Random place along radius
        
            // Calculate the coordinates of the random point
            x = center[0] + distance * Math.cos(angle); // these locations are center points of player/agent location
            y = center[1] + distance * Math.sin(angle);

        } while (x < (player.width + 50) || x > (Game.width -(player.width + 50)) || y < (player.height + 50) || y >= (Game.height - (player.height + 50)));
    
        return [x, y]; // giving center
    },
    getCivilianSpawn: function() {
        let x = 20 + (Math.floor(Math.random() * (Game.width - (20 * 3)))); // leaving 30px gap on each side
        let y = 20 + (Math.floor(Math.random() * (Game.height - (20 * 3))));
        x += 20/2; // 20 is width
        y += 20/2;
        return [x, y]; // giving center
    },
    getZombieSpawn: function() {// 500-300 = 200, 140 + 0 = 140, 140+200 = 340 . 140 + 10 = 150, 340+10 = 350. max area(140,360)
        let x = 140 + (Math.floor(Math.random() * (Game.width - 300))); // leaving 140px gap on each side
        let y = 140 + (Math.floor(Math.random() * (Game.height - 300)));
        x += 20/2; // 20 is width
        y += 20/2;
        return [x, y]; // giving center
    }
    
} 

myCanvas.width = Game.width;
myCanvas.height = Game.height;
bgCanvas.width = Game.width;
bgCanvas.height = Game.height;
Game.mapBorders = [{x: 0, y: 0},{x: Game.width, y: 0},
    {x: Game.width, y: Game.height},{x: 0, y: Game.height}]
function actionReplay() {
    Game.drawBG();
    //ctx.clearRect(0,0,Game.width,Game.height);
    let c1 = 0;
    let startFlagged = true;
    let alpha = 0.2;
    let alphaCount = 0;
    let alphaMult = 1;                                    
    let alphaBase = Math.floor(agent.maxStepCount / 9); // 7, 14, 21, 28 , 35 , 42 , 49 , 56+ (*9, 1.0 alpha)
    //let civLoc = 0;
    let zomLoc = 0;
    let agentLoc = 0;
    Game.running = true;

   function animateReplay() {
    if (!Game.running) {console.info("Replay Complete");return}
    
    ctx.clearRect(0,0,Game.width,Game.height);
    let agentX = (Game.agentMoves[agentLoc][0]); // for show path
    let agentY = (Game.agentMoves[agentLoc][1]);
    //let zomX = (Game.zombieMoves[zomLoc][0]);
    //let zomY = (Game.zombieMoves[zomLoc][1]);
    
    let terminalFlag = (Game.agentMoves[agentLoc][2]);
    
    let r,g,b;
    if (startFlagged) {
        alpha = 0.2;
        alphaCount = 0;
        alphaMult = 1;
        startFlagged = false;
        console.log("Replay: ");
    }

    if (c1 > 5) {c1 = 0}

    switch(c1){
        case 0: r = 255; g = 0; b = 0 // red
        break;
        case 1: r = 255; g = 255; b = 0 // yellow (green reserved for zombie)
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
    //ctxBG.beginPath();
    //ctxBG.arc(zomX, zomY, 2, 0, 2 * Math.PI);
    //ctxBG.fillStyle = `rgba(0,255,0,${alpha})`;
    //ctxBG.fill();

    player.x = (Game.agentMoves[agentLoc][0]) - (player.width / 2);  // getting the top left corner
    player.y = (Game.agentMoves[agentLoc][1]) - (player.height / 2); // coords are center unless referencing object
    player.draw();

    zom1.x = (Game.zombieMoves[zomLoc][0]) - (zom1.width / 2);  // getting the top left corner
    zom1.y = (Game.zombieMoves[zomLoc][1]) - (zom1.height / 2); // coords are center unless referencing object
    zom1.draw();
    
    //civ1.x = (Game.civilianMoves[civLoc][0]);
    //civ1.y = (Game.civilianMoves[civLoc][1]);
    //civ1.draw();

    if (terminalFlag) {
        startFlagged = true;
        
        c1++;
        //if ((Game.civilianMoves.length > 0) && (civLoc < Game.civilianMoves.length - 1)) {civLoc++}
    }
    agentLoc++
    zomLoc++
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
    let max = agent.maxStepCount * 25; // Max episodes (technically steps) to draw
    let alpha = 0.2;
    let alphaCount = 0;
    let alphaMult = 1;                                    
    let alphaBase = Math.floor(agent.maxStepCount / 9); // 7, 14, 21, 28 , 35 , 42 , 49 , 56+ (*9, 1.0 alpha)
    let civColors = [];

    if (Game.agentMoves.length > max) {
        start = Game.agentMoves.length - max;
    }
     
   for (let i = start; i < Game.agentMoves.length; i++) {
    let agentX = (Game.agentMoves[i][0]);
    let agentY = (Game.agentMoves[i][1]);
    //let zomX = (Game.zombieMoves[i][0]);
    //let zomY = (Game.zombieMoves[i][1]);
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
    //ctxBG.beginPath();
   // ctxBG.arc(zomX, zomY, 2, 0, 2 * Math.PI);
   // ctxBG.fillStyle = "green";
    //ctxBG.fill();

    if (terminalFlag) {
        startFlagged = true;
        let rgb = `rgb(${r},${g},${b})`;
        civColors.push(rgb);
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

// map width: 600,
// mapheight: 500,
// player: 150, 150
// civ1: 400, 150
const player = new Player(++Game.entityCounter, 150, 150, "TD3");
const zom1 = new Zombie(++Game.entityCounter, 400, 200);
//const civ1 = new Civilian(++Game.entityCounter, 400, 200);

function loadEnts() {
Game.entities.push(player); // init values
//Game.entities.push(civ1 = new Civilian(++Game.entityCounter, 400, 200));
Game.entities.push(zom1);
}

Game.init();