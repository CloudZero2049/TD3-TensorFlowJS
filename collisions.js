function collideCheck(pathEnt,AI=false){
    if (AI) {
        // actor is an array with x,y coords
        let actor = pathEnt;
        if (!Game.entities || !actor) {return}
        let ents = Game.entities;
        let colliders = [];
        for (let i = ents.length -1; i >= 0; i--) {
                if (!ents[i] || actor == ents[i]) {continue}
                ctx.beginPath;
                ents[i].addPath()
                let baseX = actor[0], baseY = actor[1], width = (actor[0] + 5),height = (actor[1] + 5);
                let x = baseX, y = baseY;

                //let signLastX = Math.sign(lastX), signLastY = Math.sign(lastY);
                //let collide = false;
                // top left
                if (ctx.isPointInPath(x,y)) {colliders.push(ents[i]);continue}
                y += height;
                // bot left
                if (ctx.isPointInPath(x,y)) {colliders.push(ents[i]);continue}
                x += width;
                y = baseY;
                // top right
                if (ctx.isPointInPath(x,y)) {colliders.push(ents[i]);continue}
                y += height;
                // bot right
                if (ctx.isPointInPath(x,y)) {colliders.push(ents[i]);continue}
        }   
        if (colliders.length > 0) {
                return colliders[0].type;
        }
        return
    }



    if (!Game.entities || !pathEnt) {return}
    let ents = Game.entities;
    let colliders = [];
    for (let i = ents.length -1; i >= 0; i--) {
      if (!ents[i] || pathEnt == ents[i]) {continue}
      let baseX = ents[i].x, baseY = ents[i].y, width = ents[i].width,height = ents[i].height, lastX = ents[i].lastMoveX, lastY = ents[i].lastMoveY;
      let x = baseX, y = baseY;
      //let signLastX = Math.sign(lastX), signLastY = Math.sign(lastY);
      //let collide = false;
      // top left
      if (ctx.isPointInPath(x,y)) {colliders.push(ents[i]);continue}
      y += height;
      // bot left
      if (ctx.isPointInPath(x,y)) {colliders.push(ents[i]);continue}
      x += width;
      y = baseY;
      // top right
      if (ctx.isPointInPath(x,y)) {colliders.push(ents[i]);continue}
      y += height;
      // bot right
      if (ctx.isPointInPath(x,y)) {colliders.push(ents[i]);continue}
    }
      if (colliders.length > 0) { //console.warn("colide");
        for (let i = colliders.length -1; i >= 0; i--) {
            calculateReward(colliders[i].type);
            Game.removeEnt(colliders[i])
        }
        /*
        let speed = ents[i].speed;
        switch (signLastX) {
          case -1: ents[i].lastMoveX = 0;
                  ents[i].x += speed;
          break;
          case 1: ents[i].lastMoveX = 0;
                  ents[i].x -= speed;
          break;
          default: // 0
          break;
        }
        switch (singLastY) {
          case -1: ents[i].lastMoveY = 0;
                  ents[i].y += speed;
          break;
          case 1: ents[i].lastMovey = 0;
                  ents[i].y -= speed;
          break;
          default: // 0
          break;
        }
        */
      }
    
  }

function calculateReward(type) { // we can move this if we have to
    switch (type) {
        case "civilian": // + 10 reward
                Game.rewards++;
                Game.updateRewardtext();
        break;
        case "zombie": // - 10 reward / punish
                Game.punishments++;
                Game.updateRewardtext();
        break;
        default: // 0 . maybe add a - 1 for walls
        break;
      }
}
