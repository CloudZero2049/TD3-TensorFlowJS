let keyTracker = {up: false,
    down: false,
    left: false,
    right: false,
    }

function moveUp() {
keyTracker.up = true;
if (keyTracker.down == true) {return}
player.lastMoveY = -(player.speed);
}

function moveDown() {
keyTracker.down = true;
if (keyTracker.up == true) {return}
player.lastMoveY = player.speed;
}

function moveLeft() {
keyTracker.left = true;
if (keyTracker.right == true) {return}
player.lastMoveX = -(player.speed);
}

function moveRight() {
keyTracker.right = true;
if (keyTracker.left == true) {return}
player.lastMoveX = player.speed;
}

function clearmove(e) {

    if (e.key == "w") {
        keyTracker.up = false;
        player.lastMoveY = 0;
        if (keyTracker.down == true) {moveDown()}
    }

    if (e.key == "s") {
        keyTracker.down = false;
        player.lastMoveY = 0;
        if (keyTracker.up == true) {moveUp()}
    }

    if (e.key == "a") {
        keyTracker.left = false;
        player.lastMoveX = 0;
        if (keyTracker.right == true) {moveRight()}
    }

    if (e.key == "d") {
        keyTracker.right = false;
        player.lastMoveX = 0;
        if (keyTracker.left == true) {moveLeft()}
    }
}



function addKeyboardListeners() {

    document.onkeydown=(e)=>{ if (!Game.user) {return}
        if (e.key !== "9" && !Game.running) {return}
        //console.log(e.key);
        switch(e.key){
            case "w": Game.user == "player" ? moveUp() : null;
            break;
            case "s": Game.user == "player" ? moveDown() : null;
            break;
            case "a": Game.user == "player" ? moveLeft() : null;
            break;
            case "d": Game.user == "player" ? moveRight() : null;
            break;
            case "0": Game.stop();
            break;
            case "9": Game.start();
            break;
        //  case "s": console.info(localStorage);
        // break; 
            // case "r": game.reset();
        // break;  
                
        }
        
    }

    document.onkeyup = function(e) {clearmove(e);}

}

addKeyboardListeners();