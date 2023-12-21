const userTrainingControl = {
    episodes: 5,
    maxStepCount: 256,
    currentEpisode: 0,
    currentStep: 0,
    newEpisode: true,
    state: observationSpace.stateSpace,
    animationTime: null,
    animateSpeed: 25,
}

function trainFromMemory() {
    if (agent.memory.cnt < 1) {return}
    const episodes = UI.episodeSlider.value;
    const steps = UI.stepSlider.value;
    console.log(`Training from memory: ${episodes} episodes with ${steps} steps...`);

  let episodesTrained = 0;
    for (let i = 0; i < episodes; i++) {
      for (let j = 0; j < steps; j++) {
        agent.train();
        
      }
      episodesTrained++;
      //if (agent.gamma > 0.2) {
      //  agent.gamma -= 0.001
      //}
      console.log(`Episodes Trained: ${episodesTrained}`);
    }
    console.log(`Finished training from memory`);
    UI.enableUI();
}

function simulateEnvStep(action, currentStep) {
    
    const OS = observationSpace;
    const osNS = observationSpace.next_stateSpace;
    const LN = observationSpace.locNums
    const scale4 = 10000
    let hitWall = false;

    const x = parseFloat(player.speed * action[0].toFixed(4)); 
    const y = parseFloat(player.speed * action[1].toFixed(4));

    // move X
    if ((OS.agentCoords[0] + x) < (player.width/2)) { // hit left wall
        OS.agentCoords[0] = (player.width/2); 
        const base = (player.width / 2) * OS.xyNorm;
        const floorX = Math.floor(base * scale4);
        osNS[0][LN.aX] = (floorX / scale4);
        player.x = 0;
        hitWall = true;
    }
    else if ((OS.agentCoords[0] + x) > (Game.width - (player.width/2))) { // hit right wall
        OS.agentCoords[0] = (Game.width - (player.width/2));
        const base = (Game.width - (player.width/2)) * OS.xyNorm;
        const floorX = Math.floor(base * scale4);
        osNS[0][LN.aX] = (floorX / scale4);

        player.x = Game.width - player.width;
        hitWall = true;
    } // player width & height = 20. x,y is center
    else {
        OS.agentCoords[0] += x;
        const base = OS.agentCoords[0] * OS.xyNorm;
        const floorX = Math.floor(base * scale4);
        osNS[0][LN.aX] = (floorX / scale4);
        player.x += x;
    }
  
    // move Y
    if ((OS.agentCoords[1] + y) < (player.height/2)) { // hit top wall
      OS.agentCoords[1] = (player.height/2);
      const base = (player.height/2) * OS.xyNorm;
      const floorY = Math.floor(base * scale4);
      osNS[0][LN.aY] = (floorY / scale4);
      player.y = 0;
      hitWall = true;
    }
    else if ((OS.agentCoords[1] + y) > (Game.height - (player.height/2))) { // hit bottom wall
      OS.agentCoords[1] = (Game.height - (player.height/2)); 
      const base = (Game.height - (player.height/2)) * OS.xyNorm;
      const floorY = Math.floor(base * scale4);
      osNS[0][LN.aY] = (floorY / scale4);
      player.y = Game.height - player.height;
      hitWall = true;
    }
    else {
      OS.agentCoords[1] += y;
      const base = OS.agentCoords[1] * OS.xyNorm;
      const floorY = Math.floor(base * scale4);
      osNS[0][LN.aY] = (floorY / scale4);
      player.y += y;
    }
    //console.log(`actionX: ${action[0]}, actionY: ${action[1]}, moveX: ${x}, moveY: ${y}`);
    //console.log(`agentCX: ${OS.agentCoords[0]}, agentCY: ${OS.agentCoords[1]}`);
    //console.log(`aX: ${osNS[0][LN.aX]}, aY: ${osNS[0][LN.aY]}, pX: ${player.x}, pY: ${player.y}`);

    const zomAction = zom1.chase(); // [x,y]
    OS.zomCoords[0] = zomAction[0];
    OS.zomCoords[1] = zomAction[1];
    //osNS[0][LN.zX] = parseFloat((zomAction[0] * OS.xyNorm).toFixed(4));
    //osNS[0][LN.zY] = parseFloat((zomAction[1] * OS.xyNorm).toFixed(4));
  
    zom1.createPolygon(); // can return the polygon points
  
    
    player.sensor.update(Game.mapBorders,[zom1]); // hardcoded zombie
    const offsets = player.sensor.readings.map((reading) => {
      if (reading == null) {
          return 1;  // Default value for no detection
      } else {
          // Adjust the offset based on the type of the detected object
          if (reading.type === "wall") {
              // Map wall distances from 1 to 0 (decending)
              return reading.offset;
          } else if (reading.type === "zombie") {
              // Map zombie distances from 0 to -1 (decending)
              return -(1 - reading.offset);
          } else {
              return reading.offset;  // Default value for unknown types
          }
      }
  });
    
    for (let i = 0; i < offsets.length; i++){
      let copy = JSON.parse(JSON.stringify(offsets[i]));
      osNS[0][i+2] = parseFloat((copy).toFixed(3));
    }
    


    let isDone = false;
    let reward = 0;
    let penalty = 0;
    
    let zomDist = utilsAI.distance(OS.agentCoords[0], OS.agentCoords[1], OS.zomCoords[0], OS.zomCoords[1]);
    //let zomAngle = utilsAI.angle([OS.agentCoords[0], OS.agentCoords[1]], [OS.zomCoords[0], OS.zomCoords[1]]); 
    
    /*
    let distancePenalty = zomDist; 
        
    let distancePenaltyScaling = observationSpace.xyNorm; // [0, 1] xyNorm is hardcoded 0.002 based on map size
    distancePenalty *= distancePenaltyScaling;
    let invDistPenalty = 1 - distancePenalty // [-1, 0]
    invDistPenalty /= 2; // [-0.5, 0] // share space
    
    penalty += invDistPenalty;
    */
    

    const safeDistance = 90; // 110 is ray length
    const survivalBonus = 2//0.1;
    const distPenalty = 1//0.5;
    /*
    function calculateTimeBonus(step, maxSteps) {
      const minBonus = 0.1; // step 0
      const maxBonus = 0.5;   // maxSteps
      const timeRatio = step / maxSteps;
      const timeBonus = minBonus + (maxBonus - minBonus) * timeRatio;
      return timeBonus;
    }
    */
    /*
    function calculateDistancePenalty() {
      const minPenalty = 0.1; // 
      const maxPenalty = 1;   // 
      const ratio = safeDistance / zomDist;
      const distPenalty = minPenalty + (maxPenalty - minPenalty) * ratio;
      return distPenalty;
    }
    */
    if (zomDist > safeDistance){ 
      //let timeBonus = calculateTimeBonus(currentStep, agent.maxStepCount);
      reward += survivalBonus; 
    }
    //else{ penalty += calculateDistancePenalty();}
    else {
      if (OS.rawDist > zomDist){ penalty += distPenalty; } // only apply distance penalty if agent isn't moving away from zombie
      //else {reward += (survivalBonus / 2)} // get a bonus for moving away from zombie
    }
    
    
    //console.log(`Distance Penalty: ${distancePenalty}`);
    //console.log(`Inverted Distance Penalty: ${invDistPenalty}`);
    //console.log(`Angle Penalty: ${anglePenalty}`);
    //console.log(`total Reward: ${totalReward}`);

    OS.rawDist = zomDist;
    //OS.rawAngle = zomAngle;
    osNS[0][LN.dist] = parseFloat((zomDist * OS.xyNorm).toFixed(3));
    //osNS[0][LN.angle] = parseFloat((zomAngle / Math.PI).toFixed(4));
    //osNS[0][LN.angle] = (zomAngle + Math.PI) / (2 * Math.PI); // [0,1]
    // const timeBonus = calculateTimeBonus(currentStep, agent.maxStepCount);
    //reward += timeBonus;
    
    //if (penalty > 1) {penalty = 1} // failsafe
    //if (reward > 1) {reward = 1} // failsafe
    
    if (hitWall) { // huge penalty for walking into walls
   
      penalty += 1//0.5; // walking into walls is bad
      
    } 
    if (penalty > 2) {penalty = 2} // failsafe
    if (reward > 2) {reward = 2} // failsafe
    reward = reward - penalty;

    if (zomDist <= player.width) { // If zombie finds agent, is very bad
      reward = -2; 
      
      isDone = true;
    }
    else if (currentStep >= agent.maxStepCount) {
      //reward = 1;
      isDone = true;
    }
    
    const next_state = JSON.parse(JSON.stringify(observationSpace.next_stateSpace));
    //console.log(next_State);
   
    return { next_state: next_state, reward: reward, isDone: isDone };
  }


  function animate() {
    if (!Game.running) {console.info("Animation Stopped");return}
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
    aX = coinFlip > 0.5 ? 0.1 : -0.1; // Make sure this is below agent movement threshold
  }
  if (aY == 0) {
    let coinFlip = Math.random();
    aY = coinFlip > 0.5 ? 0.1 : -0.1;
  }
  const safeAction = [aX, aY];
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