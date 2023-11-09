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


let player;

let Game = {
    running: false, 
    user: "TD3",
    width: 1000,
    height: 1000,
    entities: [],
    player: {},
    entityCounter: 0,
    rewards: 0,
    punishments: 0,
    init: function() {
        this.drawBG();
        loadEnts();
        this.updateRewardtext();
        for (let i = this.entities.length -1; i >= 0; i--) {
            if (!this.entities[i]) {continue}
            this.entities[i].draw();
        }
        // Dummy Player Placeholder
        ctx.beginPath;
        ctx.fillStyle = "blue";
        ctx.fillRect(50,50,20,20);

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
            main()
        }
        else if(this.user == "player") {
            this.loadPlayer();
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
        Game.entities.push(player = new Player(++Game.entityCounter,50,50,"human"));
    }
} 

myCanvas.width = 600;
myCanvas.height = 600;
bgCanvas.width = 600;
bgCanvas.height = 600;

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


function loadEnts() {

Game.entities.push(new Civilian(++Game.entityCounter,90,90));
Game.entities.push(new Zombie(++Game.entityCounter,10,10));
//zombies.push(new Zombie(++game.entityCounter,300,300,"medium"));
}

Game.init();
