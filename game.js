//GAME INFO: the Twin Delayed Deep Deterministic Policy Gradient (TD3) agent will be given inputs (x,y coordinates of self, zombies, and civilians), and will have outputs (moving across x,y plane)
// Agent will get rewards for touching civilians, and punishment for touching zombies.

window.ondrag = function(e) {e.preventDefault();};
window.ondblclick = function(e) {e.preventDefault();};
window.oncontextmenu = function(e) {e.preventDefault();};

const startPlayerButton = document.getElementById("startPlayerButton");
const startTD3Button = document.getElementById("startTD3Button");

const ctx = myCanvas.getContext("2d");
const ctxBG = bgCanvas.getContext("2d");
const rewardTextArea = document.getElementById("rewardTextArea");
const punishTextArea = document.getElementById("punishTextArea");


let player, civ1;

let Game = {
    running: false, 
    user: "TD3",
    width: 600,
    height: 500,
    entities: [],
    player: {},
    entityCounter: 0,
    rewards: 0,
    punishments: 0,
    agentMoves: [],
    init: function() {
        this.drawBG();
        loadEnts();
        this.updateRewardtext();
        for (let i = this.entities.length -1; i >= 0; i--) {
            if (!this.entities[i]) {continue}
            this.entities[i].draw();
        }
        // Dummy Player Placeholder
        //ctx.beginPath;
       // ctx.fillStyle = "blue";
       // ctx.fillRect(150,150,20,20);

        startPlayerButton.disabled = false;
        startTD3Button.disabled = false;
        //this.start();
    },
    start: function(user) {
        if (user) {
            this.user = user == `player` ? "player" : "TD3";
        }
        if (this.running) {return}
        this.running = true;       
        startPlayerButton.disabled = true;
        startTD3Button.disabled = true;
        
        if (this.user == "TD3") {
            player.user = "TD3";
            main()
        }
        else if(this.user == "player") {
            player.user = "human";
            animate(); 
        }
        console.info(tf.memory());
        console.info("TD3 Complete");
        startTD3Button.disabled = false;
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
        ctxBG.beginPath;
        ctxBG.fillStyle = "rgb(204, 255, 221)";
        ctxBG.fillRect(0,0,bgCanvas.width,bgCanvas.height);
    },
    updateRewardtext: function() {
        rewardTextArea.innerHTML = `Rewards: ${this.rewards}`;
        punishTextArea.innerHTML = `Punishments: ${this.punishments}`;
    },
    loadPlayer: function() {
       // Game.entities.push(player = new Player(++Game.entityCounter,150,150,"human"));
    },
    loadAgent: function() {
       // Game.entities.push(player = new Player(++Game.entityCounter,150,150,"TD3"));
    }
} 

myCanvas.width = Game.width;
myCanvas.height = Game.height;
bgCanvas.width = Game.width;
bgCanvas.height = Game.height;

function animate() {
    if (!Game.running) {console.info("Game Stopped");return}
    let ents = Game.entities;
    ctx.clearRect(0,0,Game.width,Game.height);
    
    if (Game.user == "TD3") {   // Agent controled
        for (let i = ents.length -1; i >= 0; i--) {
            if (!ents[i]) {continue}
            ents[i].draw();
        }
    }
    else {  // human controlled
        ctx.beginPath;
        player.addPath();
        collideCheck(player)

        for (let i = ents.length -1; i >= 0; i--) {
            if (!ents[i]) {continue}
            
            if (ents[i].lastMoveX != 0 || ents[i].lastMoveY != 0) {
                ents[i].x += ents[i].lastMoveX;
                ents[i].y += ents[i].lastMoveY;     
            }

            ents[i].draw();

        }
    }
    requestAnimationFrame(animate);
   
}
function animateAgent() {
    ctx.clearRect(player.x-1,player.y-1,player.width+2,player.height+2);
    let c1 = 0;
    let c2 = 0;
   for (let i = 0; i < Game.agentMoves.length; i++) {
    let moveX = Game.agentMoves[i][0];
    let moveY = Game.agentMoves[i][1];
    //console.log(`moveX: ${moveX}, moveY: ${moveY }`);
    //let r = Math.floor(Math.random() * (255 + 1))
    //let g = Math.floor(Math.random() * (255 + 1))
    //let b = Math.floor(Math.random() * (255 + 1))
    let r,g,b;
    
    switch(c1){
        case 0: r = 255; g = 0; b = 0
        break;
        case 1: r = 0; g = 255; b = 0
        break;    
        case 2: r = 0; g = 0; b = 255
        break; 
      }

    let rgb = `rgb(${r},${g},${b})`;
    c2++
    if (c2 > agent.batch_size) {c1++; c2 = 0;}
    if (c1 > 2) {c1 = 0}

    ctxBG.beginPath();
    ctxBG.arc(moveX, moveY, 2, 0, 2 * Math.PI);
    ctxBG.fillStyle = rgb;
    ctxBG.fill();

   }

    player.x = Game.agentMoves[Game.agentMoves.length -1][0];
    player.y = Game.agentMoves[Game.agentMoves.length -1][1];
    
    player.draw();
}
// map width: 600,
// mapheight: 500,
// player: 150, 150
// civ1: 400, 150

function loadEnts() {
Game.entities.push(player = new Player(++Game.entityCounter, 300, 50, "TD3"));
Game.entities.push(civ1 = new Civilian(++Game.entityCounter, 200, 200));
Game.entities.push(new Zombie(++Game.entityCounter,10,10));
//zombies.push(new Zombie(++game.entityCounter,300,300,"medium"));
}

Game.init();
