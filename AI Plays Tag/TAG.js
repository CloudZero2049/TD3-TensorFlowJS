const tagControl = {
    episodes: 5,
    maxStepCount: 128,
    currentEpisode: 0,
    currentStep: 0,
    newEpisode: true,
    animationTime: null,
    animateSpeed: 20,
    chaserSpeed: 3,
    runnerSpeed: 5,
    stunTimer: 500,
}

function tagStep() {
    
    const OS = observationSpace;
    const LN = observationSpace.locNums
    const ents = Game.entities;

    const movementThreshold = 0;
  // move X
  for (let i = ents.length -1; i >= 0; i--) { // movement
    if(ents[i].stunned){continue}

    let actionClone = JSON.parse(JSON.stringify(ents[i].action));
    let osNS = ents[i].nextStateSpace;
    let actX = actionClone[0];
    let actY = actionClone[1];
    let x = ents[i].speed * actionClone[0];
    let y = ents[i].speed * actionClone[1];
    
    ctx.beginPath();
    for (let j = ents.length -1; j >= 0; j--) {
      if (!ents[j] || j == i) {continue}
      if (ents[j].role != ents[i].role || (ents[j].role == "chaser" && ents[i].role == "chaser")) {continue}
      ents[j].addPath();
    }
    let colide = Game.colideCheckIPIP(ents[i],x,y);
    
    if (!colide) {
      if ((actX < -movementThreshold) || (actX > movementThreshold)) {
        if ((ents[i].location[0] + x) < (ents[i].width/2)) { // hit left wall
          ents[i].location[0] = (ents[i].width/2); 
          const base = (ents[i].width / 2) * OS.xyNorm;
          const floorX = Math.floor(base * agent.decScale);
          osNS[0][LN.aX] = (floorX / agent.decScale);    
          ents[i].x = 0;
        }
        else if ((ents[i].location[0] + x) > (Game.width - (ents[i].width/2))) { // hit right wall
          ents[i].location[0] = (Game.width - (ents[i].width/2));
          const base = (Game.width - (ents[i].width/2)) * OS.xyNorm;
          const floorX = Math.floor(base * agent.decScale);
          osNS[0][LN.aX] = (floorX / agent.decScale);
          ents[i].x = Game.width - ents[i].width;
        } // ents[i] width & height = 20. x,y is center
        else {
          ents[i].location[0] += x;
          const base = ents[i].location[0] * OS.xyNorm;
          const floorX = Math.floor(base * agent.decScale);
          osNS[0][LN.aX] = (floorX / agent.decScale);
          ents[i].x += x;
        }
      } // end move x
  
    // move Y
    if ((actY < -movementThreshold) || (actY > movementThreshold)) {
      if ((ents[i].location[1] + y) < (ents[i].height/2)) { // hit top wall
        ents[i].location[1] = (ents[i].height/2);
        const base = (ents[i].height/2) * OS.xyNorm;
        const floorY = Math.floor(base * agent.decScale);
        osNS[0][LN.aY] = (floorY / agent.decScale);
        ents[i].y = 0;
      }
      else if ((ents[i].location[1] + y) > (Game.height - (ents[i].height/2))) { // hit bottom wall
        ents[i].location[1] = (Game.height - (ents[i].height/2)); 
        const base = (Game.height - (ents[i].height/2)) * OS.xyNorm;
        const floorY = Math.floor(base * agent.decScale);
        osNS[0][LN.aY] = (floorY / agent.decScale);
        ents[i].y = Game.height - ents[i].height; 
      }
      else {
        ents[i].location[1] += y;
        const base = ents[i].location[1] * OS.xyNorm;
        const floorY = Math.floor(base * agent.decScale);
        osNS[0][LN.aY] = (floorY / agent.decScale);
        ents[i].y += y;
      }
    } // end move Y
  } // end if not colide
} // end ent loop

  for (let i = ents.length -1; i >= 0; i--) {
    if (!ents[i]) {continue}
    ents[i].createPolygon();
  }
    
  for (let i = ents.length -1; i >= 0; i--) {
    if (!ents[i]) {continue}
    ents[i].role == "chaser" ?  ents[i].updateChaserSensor() : ents[i].updateRunnerSensor();
  }
    
    let isDone = false;
    
    for (let i = ents.length -1; i >= 0; i--) {
      if (!ents[i]) {continue}
      for (let j = i - 1; j >= 0; j--) {
        if (!ents[j] || ents[j].role == ents[i].role) {continue}

        let distance = utilsAI.distance(ents[i].location[0], ents[i].location[1], ents[j].location[0], ents[j].location[1]);

        if (distance <= 20 && !ents[i].stunned && !ents[j].stunned) {
          if (ents[i].role == "chaser"){
            ents[i].tags++;
            ents[i].role = "runner";
            ents[j].role = "chaser";

            ents[j].stunned = true;
            ents[j].speed = tagControl.chaserSpeed;
            ents[i].speed = tagControl.runnerSpeed;
            setTimeout(() => {ents[j].stunned = false;}, tagControl.stunTimer);
          }
          else {
            ents[j].tags++;
            ents[i].role = "chaser";
            ents[j].role = "runner";

            ents[i].stunned = true;
            ents[i].speed = tagControl.chaserSpeed;
            ents[j].speed = tagControl.runnerSpeed;
            setTimeout(() => {ents[i].stunned = false;}, tagControl.stunTimer);
          }
          
          ents[i].color = Game.getColor(ents[i].role);
          ents[j].color = Game.getColor(ents[j].role);
          
        } // end if tagged
      }
  }

    return {isDone: isDone};
  }

  function animatetag() {
    if (!Game.running) {
      clearInterval(tagControl.animationTime);
      return
    }
    
    const ents = Game.entities;
    const MTC = tagControl;
    
    if (MTC.newEpisode) {
      envReset();
      
      MTC.currentStep = 0;
      MTC.newEpisode = false;
      console.log(`Episode ${MTC.currentEpisode + 1}`);
    }
    
    for (let i = ents.length -1; i >= 0; i--) { // Choose actions
      if (!ents[i]) {continue}
      
      let action;
      if(ents[i].role == "chaser") {
        action = agent.act(ents[i].stateSpace, agent.chaser_actor);
      }
      else {
        action = agent.act(ents[i].stateSpace, agent.runner_actor);
      }
      let aX = action[0]; 
      let aY = action[1];
      
      if (action[0] > -1 && action[0] < 1){
        aX = Math.floor(action[0] * agent.decScale); 
        aX = (aX / agent.decScale);
      }
      if (action[1] > -1 && action[1] < 1){
        aY = Math.floor(action[1] * agent.decScale);
        aY = (aY / agent.decScale);
      }

      ents[i].action = [aX,aY];

    } // end ent loop

    const {isDone} = tagStep(MTC.currentStep); // Take actions

    ctx.clearRect(0,0,Game.width,Game.height);
    for (let i = ents.length -1; i >= 0; i--) {
      if (!ents[i]) {continue}
      ents[i].draw();
      ents[i].stateSpace = JSON.parse(JSON.stringify(ents[i].nextStateSpace));
    }
    for (let i = ents.length -1; i >= 0; i--) {
      if (!ents[i] || ents[i].role == "chaser") {continue}
      ents[i].runTime++;
    }
    UI.updatePlayersTable();
    MTC.currentStep++;
    
    if (isDone|| MTC.currentStep >= MTC.maxStepCount) {
        MTC.newEpisode = true;
        MTC.currentEpisode++
    }
    if(MTC.currentEpisode >= MTC.episodes) {
        Game.running = false;
        UI.updatePlayersTable(true);
        console.log("Tag Complete");
        
        clearInterval(tagControl.animationTime);
        UI.enableUI();
    }
   
}

  function tagMain(epNum,stepSize) {
   
    const MTC = tagControl;
    if (epNum && !isNaN(epNum) && epNum > 0) {MTC.episodes = epNum}
    if (stepSize && !isNaN(stepSize) && stepSize > 0) {MTC.maxStepCount = stepSize}
    
    MTC.currentStep = 0;
    MTC.currentEpisode = 0;
    MTC.newEpisode = true;
    const ents = Game.entities;
    for (let i = ents.length -1; i >= 0; i--) {
      if(!ents[i]) {continue}
      ents[i].speed = ents[i].role == "chaser" ? MTC.chaserSpeed : MTC.runnerSpeed;
    }

    MTC.animationTime = setInterval(animatetag, MTC.animateSpeed);
  }