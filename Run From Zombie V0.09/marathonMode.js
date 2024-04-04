const marathonControl = {
    episodes: 5,
    maxStepCount: 128,
    currentEpisode: 0,
    currentStep: 0,
    newEpisode: true,
    state: observationSpace.stateSpace,
    animationTime: null,
    animateSpeed: 20,
}

function marathonStep(action, currentStep) {
    
    const OS = observationSpace;
    const osNS = observationSpace.next_stateSpace;
    const LN = observationSpace.locNums
    const actionClone = JSON.parse(JSON.stringify(action));
    const ents = Game.entities;
    let hitWall = false;

    const x = player.speed * actionClone[0]; 
    const y = player.speed * actionClone[1];

    const movementThreshold = 0; // 0.15 then 0.05
  // move X
  if ((actionClone[0] < -movementThreshold) || (actionClone[0] > movementThreshold)) {
    if ((OS.agentCoords[0] + x) < (player.width/2)) { // hit left wall
      OS.agentCoords[0] = (player.width/2); 
      const base = (player.width / 2) * OS.xyNorm;
      const floorX = Math.floor(base * agent.decScale);
      osNS[0][LN.aX] = (floorX / agent.decScale);    
      player.x = 0;
      hitWall = true;
    }
    else if ((OS.agentCoords[0] + x) > (Game.width - (player.width/2))) { // hit right wall
      OS.agentCoords[0] = (Game.width - (player.width/2));
      const base = (Game.width - (player.width/2)) * OS.xyNorm;
      const floorX = Math.floor(base * agent.decScale);
      osNS[0][LN.aX] = (floorX / agent.decScale);
      player.x = Game.width - player.width;
      hitWall = true;
    } // player width & height = 20. x,y is center
    else {
      OS.agentCoords[0] += x;
      const base = OS.agentCoords[0] * OS.xyNorm;
      const floorX = Math.floor(base * agent.decScale);
      osNS[0][LN.aX] = (floorX / agent.decScale);
      player.x += x;
    }
  }
  
  // move Y
  if ((actionClone[1] < -movementThreshold) || (actionClone[1] > movementThreshold)) {
    if ((OS.agentCoords[1] + y) < (player.height/2)) { // hit top wall
      OS.agentCoords[1] = (player.height/2);
      const base = (player.height/2) * OS.xyNorm;
      const floorY = Math.floor(base * agent.decScale);
      osNS[0][LN.aY] = (floorY / agent.decScale);
      player.y = 0;
      hitWall = true;
    }
    else if ((OS.agentCoords[1] + y) > (Game.height - (player.height/2))) { // hit bottom wall
      OS.agentCoords[1] = (Game.height - (player.height/2)); 
      const base = (Game.height - (player.height/2)) * OS.xyNorm;
      const floorY = Math.floor(base * agent.decScale);
      osNS[0][LN.aY] = (floorY / agent.decScale);
      player.y = Game.height - player.height; 
      hitWall = true;
    }
    else {
      OS.agentCoords[1] += y;
      const base = OS.agentCoords[1] * OS.xyNorm;
      const floorY = Math.floor(base * agent.decScale);
      osNS[0][LN.aY] = (floorY / agent.decScale);
      player.y += y;
    }
  }
   
  for (let i = ents.length -1; i >= 0; i--) {
    if (!ents[i] || ents[i].type == "player") {continue}
    ents[i].createPolygon();
  }
    //civ1.createPolygon(); // can return the polygon points
  
    player.sensor.update(Game.mapBorders,Game.entities); // hardcoded zombie
    const offsets = player.sensor.readings.map((reading) => {
      if (reading == null) {
          return 1;  // Default value for no detection
      } else {
          // Adjust the offset based on the type of the detected object
          if (reading.type === "civilian") {
              // Map distances from 0 to 1 (ascending)
              //return 1 - reading.offset;
              return parseFloat(reading.offset.toFixed(4));
          } else if (reading.type === "wall") { // disabled
              // Map distances from 0 to -1 (decending)
              //return parseFloat((-(1 - reading.offset)).toFixed(3));
          } else {
              return parseFloat(reading.offset.toFixed(4)); // Default value for unknown types
          }
      }
  });
  
  let detect = false;

  for (let i = 0; i < offsets.length; i++){
    let copy = JSON.parse(JSON.stringify(offsets[i]));
    const clipCopy = parseFloat((copy).toFixed(4));
    if (clipCopy > 0 && clipCopy < 1) {detect = true}
    //if (clipCopy < 0.4 && clipCopy > 0 && clipCopy < osNS[0][i+2]) {closeWall = true;}
    osNS[0][i+2] = clipCopy;
    
    }

    let isDone = false;
 
  for (let i = ents.length-1; i >= 0; i--) {
    if (ents[i].type == "player") {continue}
    let cx = ents[i].x + 10;
    let cy = ents[i].y + 10;
    let dist = utilsAI.distance(OS.agentCoords[0], OS.agentCoords[1], cx, cy);

    if (dist <= player.width) {
      let civLoc = Game.getCivilianSpawn();
      let x = civLoc[0] - 10;
      let y = civLoc[1] - 10;
      ents[i].x = x;
      ents[i].y = y;
      Game.winsThisRun++;
      UI.winsThisRun.innerHTML = `Wins This Run: ${Game.winsThisRun}`;
    }
  }
    
     if (currentStep >= marathonControl.maxStepCount) {
      
      isDone = true;
    }
    

    let terminalFlag = false;
    if (isDone) {terminalFlag = true;}
    const aX = OS.agentCoords[0];
    const aY = OS.agentCoords[1];
    //const zX = OS.zomCoords[0];
    //const zY = OS.zomCoords[1];
    Game.agentMoves.push([aX, aY, terminalFlag]); // For drawing agent paths

   
    return {isDone: isDone};
  }


  function animateMarathon() {
    if (!Game.running) {
      console.info("Animation Stopped");
      clearInterval(marathonControl.animationTime);
      return
    }
    const ents = Game.entities;
    ctx.clearRect(0,0,Game.width,Game.height);

    const MTC = marathonControl;
    
    if (MTC.newEpisode) {
        envReset();
        MTC.state = JSON.parse(JSON.stringify(observationSpace.stateSpace));
        MTC.currentStep = 0;
        MTC.newEpisode = false;
        console.log("Episode ");
    }
    
    const action = agent.act(MTC.state);
    let x = action[0];
    let y = action[1];
    if (action[0] > -1 && action[0] < 1){
      x = Math.floor(action[0] * agent.decScale); // multiply by 10,000 to get a 4 digit number, then floor it
      x = (x / agent.decScale); // return to float value.
    }
    if (action[1] > -1 && action[1] < 1){
      y = Math.floor(action[1] * agent.decScale);
      y = (y / agent.decScale);
    }
    
    const clipAction = [x,y];

    const {isDone} = marathonStep(clipAction, MTC.currentStep);

    for (let i = ents.length -1; i >= 0; i--) {
      if (!ents[i]) {continue}
      ents[i].draw();
  }
  
    //console.log(`state: ${MTC.state}`);
    //console.log(`next_state: ${next_state}`);
    //console.log(`isDone: ${isDone}`);

    MTC.state = JSON.parse(JSON.stringify(observationSpace.next_stateSpace)); // DEEP COPY
    MTC.currentStep++;
    
    if (isDone) {
        MTC.newEpisode = true;
        MTC.currentEpisode++
    }
    if(MTC.currentEpisode == MTC.episodes) {
        Game.running = false;
        Game.testing = false;
        Game.marathon = false;
        console.log("Marathon Testing Complete");
        UI.trainFromMemoryButton.disabled = false;
        clearInterval(marathonControl.animationTime);
        UI.enableUI();
    }
   
}

  function marathonMain(epNum,stepSize) {
   
    const MTC = marathonControl;
    if (epNum && !isNaN(epNum) && epNum > 0) {MTC.episodes = epNum}
    if (stepSize && !isNaN(stepSize) && stepSize > 0) {MTC.maxStepCount = stepSize}
    //if (batchSize && !isNaN(batchSize) && batchSize > 0) {agent.batch_size = batchSize}

    //observationSpace.initUpdate()
    MTC.currentStep = 0;
    MTC.currentEpisode = 0;
    MTC.newEpisode = true;

    MTC.animationTime = setInterval(animateMarathon, MTC.animateSpeed);
  }