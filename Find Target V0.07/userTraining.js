const userTrainingControl = {
    episodes: 5,
    maxStepCount: 128,
    currentEpisode: 0,
    currentStep: 0,
    newEpisode: true,
    state: observationSpace.stateSpace,
    animationTime: null,
    animateSpeed: 25,
}

function trainFromMemory() {
    if (agent.memory.cnt < 1) {return}
    const episodes = parseInt(UI.episodeSlider.value);
    const steps = parseInt(UI.stepSlider.value);
    const gamma = parseFloat(UI.gammaSlider.value);
    agent.gamma = gamma;
    console.log(`Training from memory: ${episodes} episodes with ${steps} steps...`);

  let episodesTrained = 0;
    for (let i = 0; i < episodes; i++) {
      for (let j = 0; j < steps; j++) {
        agent.train();
        
      }
      episodesTrained++;
      
      console.log(`Episodes Trained: ${episodesTrained}`);
    }
    console.log(`Finished training from memory`);
    UI.enableUI();
}

function simulateEnvStep(action, currentStep) {
    
    const OS = observationSpace;
    const osNS = observationSpace.next_stateSpace;
    const LN = observationSpace.locNums
    //const decScale = 100000
    //let closeWall = false;
    let hitWall = false;

    const x = parseFloat(player.speed * action[0].toFixed(4)); 
    const y = parseFloat(player.speed * action[1].toFixed(4));

    // move X
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
  
    // move Y
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
    //console.log(`actionX: ${action[0]}, actionY: ${action[1]}, moveX: ${x}, moveY: ${y}`);
    //console.log(`agentCX: ${OS.agentCoords[0]}, agentCY: ${OS.agentCoords[1]}`);
    //console.log(`aX: ${osNS[0][LN.aX]}, aY: ${osNS[0][LN.aY]}, pX: ${player.x}, pY: ${player.y}`);

    //const zomAction = zom1.chase(); // [x,y]
    //OS.zomCoords[0] = zomAction[0];
    //OS.zomCoords[1] = zomAction[1];
    //osNS[0][LN.zX] = parseFloat((zomAction[0] * OS.xyNorm).toFixed(4));
    //osNS[0][LN.zY] = parseFloat((zomAction[1] * OS.xyNorm).toFixed(4));
  
    civ1.createPolygon(); // can return the polygon points
  
    
    player.sensor.update(Game.mapBorders,[civ1]); // hardcoded zombie
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
    let reward = 0;
    let penalty = 0;
    
    //let zomDist = utilsAI.distance(OS.agentCoords[0], OS.agentCoords[1], OS.zomCoords[0], OS.zomCoords[1]);
    //let zomAngle = utilsAI.angle([OS.agentCoords[0], OS.agentCoords[1]], [OS.zomCoords[0], OS.zomCoords[1]]); 
    let civDist = utilsAI.distance(OS.agentCoords[0], OS.agentCoords[1], OS.civCoords[0], OS.civCoords[1]);
    let civAngle = utilsAI.angle([OS.agentCoords[0], OS.agentCoords[1]], [OS.civCoords[0], OS.civCoords[1]]);
    let agentHeading = utilsAI.angle([OS.stateSpace[0][LN.aX], OS.stateSpace[0][LN.aY]], [OS.agentCoords[0], OS.agentCoords[1]]); 
   
  function angleDifferenceRadians(angle1, angle2) {
    let diff = Math.abs(angle1 - angle2);
    return Math.min(diff, 2 * Math.PI - diff);
}

// ChatGPT says that using agentHeading is better to get it to turn towards civ, otherwise it measures change over time.
let angleRadDif = angleDifferenceRadians(OS.rawAngle, agentHeading); // angleRadDif < ? or 0 + reward [0, PI]

   //1rad × 180/π = 57.296°
  //1° × π/180 = 0.01745rad
 
  let anglePenalty = angleRadDif / Math.PI; // Penalty decreases as this approaches 0. [0, 1]
  let angleBonus = 1 - angleRadDif / Math.PI;
  //anglePenalty /= 2; // [0, 0.5] // Share space with distancePenalty
  
  let distancePenalty = civDist;
  let distanceBonus = 1 / civDist;

  let distancePenaltyScaling = observationSpace.xyNorm; // [0, 1] xyNorm is hardcoded 0.002 based on map size

  distancePenalty *= distancePenaltyScaling;
  distanceBonus *= distanceScaling;
 
  distancePenalty = Math.min(distancePenalty, 1); // safegards
  distanceBonus = Math.min(distanceBonus, 1);
  
 
function calculateTimePenalty(step, maxSteps) {
  const maxPenalty = -0.2; // Adjust as needed
  const timeRatio = step / maxSteps;
  const timePenalty = Math.max(maxPenalty, maxPenalty * timeRatio);
  return timePenalty;
}

if (civDist >= OS.rawDist) {
  penalty += distancePenalty;
  penalty += anglePenalty;
  //console.log(`distancePenalty: ${distancePenalty}`);
  //console.log(`anglePenalty: ${anglePenalty}`);
}
else {reward += distanceBonus;
  //console.log(`distanceBonus: ${distanceBonus}`);
  if (anglePenalty > OS.rawAngleDiff){
    penalty += anglePenalty;
    //console.log(`anglePenalty: ${anglePenalty}`);
  }
  else if (anglePenalty < 0.1) {
    reward += angleBonus;
    //console.log(`angleBonus: ${angleBonus}`);
  } // more incentive for perfect angle
  else {
    reward += Math.max(0, angleBonus - 0.1);
    //console.log(`mod-angleBonus: ${Math.max(0, angleBonus - 0.1)}`);
  }

}

  OS.rawDist = civDist;
  OS.rawAngle = civAngle;
  OS.rawAngleDiff = anglePenalty;
  //OS.rawAngle = zomAngle;

  const timePenalty = calculateTimePenalty(currentStep, agent.maxStepCount);

  
  if (penalty > 2) {penalty = 2} // failsafes
  if (reward > 2) {reward = 2}

  if (detect) { // only give rewards and penalties if ray detect something.
    reward = reward - penalty;
  }
  else {
    reward = 0;
    penalty = 0;
  }

  if (hitWall) { // penalty for walking into walls
    reward -= 0.5;
    //isDone = true; // end if not testing
  } 
  
  reward -= Math.abs(timePenalty);
  if (reward < -2) {reward = -2}
  
  /*
  if (zomDist <= player.width) { // If zombie finds agent, is very bad
    reward = -2; 
    isDone = true;
  }
  */
    if (civDist <= player.width) { // If zombie finds agent, is very bad
      reward = 2; 
      isDone = true;
    }
    else if (currentStep >= userTrainingControl.maxStepCount) {
      
      isDone = true;
    }
    
    const next_state = JSON.parse(JSON.stringify(observationSpace.next_stateSpace));
    //console.log(next_State);
   
    return { next_state: next_state, reward: reward, isDone: isDone };
  }


  function animate() {
    if (!Game.running) {
      console.info("Animation Stopped");
      clearInterval(userTrainingControl.animationTime);
      return
    }
    let ents = Game.entities;
    ctx.clearRect(0,0,Game.width,Game.height);

    const UTC = userTrainingControl;
    
    if (UTC.newEpisode) {
        envReset();
        UTC.state = JSON.parse(JSON.stringify(observationSpace.stateSpace));
        UTC.currentStep = 0;
        UTC.newEpisode = false;
        console.log("Episode ");
    }
   
    let aX = player.lastMoveX;
    let aY = player.lastMoveY;
    
    const action = [aX, aY]
    const {next_state, reward, isDone} = simulateEnvStep(action, UTC.currentStep);

    for (let i = ents.length -1; i >= 0; i--) {
      if (!ents[i]) {continue}
      ents[i].draw();
  }
  if (aX == 0) {      // Keeping 0s out of the equations
    let coinFlip = Math.random();
    aX = coinFlip > 0.5 ? 0.0001 : -0.0001;
  }
  if (aY == 0) {
    let coinFlip = Math.random();
    aY = coinFlip > 0.5 ? 0.0001 : -0.0001;
  }
  const safeAction = [aX, aY]; // proably needs noise because [-1,1]
    //console.log(`state: ${UTC.state}`);
    //console.log(`next_state: ${next_state}`);
    //console.log(`safeAction: ${safeAction}`);
    //console.log(`isDone: ${isDone}`);
    //console.log(`reward: ${reward}`);
    agent.savexp(UTC.state, next_state, safeAction, isDone, reward);

    UTC.state = JSON.parse(JSON.stringify(observationSpace.next_stateSpace)); // DEEP COPY
    UTC.currentStep++;
    
    if (isDone) {
        UTC.newEpisode = true;
        UTC.currentEpisode++
    }
    if(UTC.currentEpisode == UTC.episodes) {
        Game.running = false;
        console.log("User Training Complete");
        UI.trainFromMemoryButton.disabled = false;
        clearInterval(userTrainingControl.animationTime);
        UI.enableUI();
    }
    //requestAnimationFrame(animate);
   
}

  function userTrainMain(epNum,stepSize,batchSize) {
    console.log("User Training Started");
    const UTC = userTrainingControl;
    if (epNum && !isNaN(epNum) && epNum > 0) {UTC.episodes = epNum}
    if (stepSize && !isNaN(stepSize) && stepSize > 0) {UTC.maxStepCount = stepSize}
    if (batchSize && !isNaN(batchSize) && batchSize > 0) {agent.batch_size = batchSize}

    //observationSpace.initUpdate()
    UTC.currentStep = 0;
    UTC.currentEpisode = 0;
    UTC.newEpisode = true;
    UTC.animationTime = setInterval(animate, UTC.animateSpeed);
    //animate(); clearInterval(userTrainingControl.animationTime);
  }