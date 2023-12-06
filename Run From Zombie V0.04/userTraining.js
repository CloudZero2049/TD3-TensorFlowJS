const userTrainingControl = {
    episodes: 5,
    maxStepCount: 256,
    currentEpisode: 0,
    currentStep: 0,
    newEpisode: true,
    state: observationSpace.stateSpace,
    animationTime: null,
    animateSpeed: 35,
}

function trainFromMemory() {
    if (agent.memory.cnt < 1) {return}
    console.log("Training from memory...");
    const episodes = UI.episodeSlider.value;

    for (let i = 0; i < episodes; i++) {
        agent.train();
    }
    console.log(`Trained ${episodes} episodes`);
}

function simulateEnvStep(action, currentStep) {
    
    const OS = observationSpace;
    const osNS = observationSpace.next_stateSpace;
    const LN = observationSpace.locNums
    let hitWall = false;
  
    zom1.createPolygon(); // can return the polygon points
  
    if(player.sensor){
      player.sensor.update(Game.mapBorders,[zom1]); // hardcoded zombie
      const offsets = player.sensor.readings.map(
          s=>s==null?-1:1-s.offset
      );
      for (let i = 0; i < offsets.length; i++){
        osNS[0][i+6] = offsets[i];
      }
    }


    const x = (player.speed * action[0]); 
    const y = (player.speed * action[1]);

    // move X
    if ((OS.agentCoords[0] + x) < (player.width/2)) { // hit left wall
        OS.agentCoords[0] = (player.width/2); 
        osNS[0][LN.aX] = (player.width/2) * OS.xyNorm;
        player.x = 0;
        hitWall = true;
    }
    else if ((OS.agentCoords[0] + x) > (Game.width - (player.width/2))) { // hit right wall
        OS.agentCoords[0] = (Game.width - (player.width/2));
        osNS[0][LN.aX] = (Game.width - (player.width/2)) * OS.xyNorm;
        player.x = Game.width - player.width;
        hitWall = true;
    } // player width & height = 20. x,y is center
    else {
        OS.agentCoords[0] += x;
        osNS[0][LN.aX] += x * OS.xyNorm;
        player.x += x;
    }
  
    // move Y
    if ((OS.agentCoords[1] + y) < (player.height/2)) { // hit top wall
      OS.agentCoords[1] = (player.height/2);
      osNS[0][LN.aY] = (player.height/2) * OS.xyNorm;
      player.y = 0;
      hitWall = true;
    }
    else if ((OS.agentCoords[1] + y) > (Game.height - (player.height/2))) { // hit bottom wall
      OS.agentCoords[1] = (Game.height - (player.height/2)); 
      osNS[0][LN.aY] = (Game.height - (player.height/2)) * OS.xyNorm;
      player.y = Game.height - player.height;
      hitWall = true;
    }
    else {
      OS.agentCoords[1] += y;
      osNS[0][LN.aY] += y * OS.xyNorm;
      player.y += y;
    }
   
    const zomAction = zom1.chase(); // [x,y]
    OS.zomCoords[0] = zomAction[0];
    OS.zomCoords[1] = zomAction[1];
    osNS[0][LN.zX] = zomAction[0] * OS.xyNorm;
    osNS[0][LN.zY] = zomAction[1] * OS.xyNorm;;
  
    let isDone = false;
    let reward = 0;
    let penalty = 0;
    
    let zomDist = utilsAI.distance(OS.agentCoords[0], OS.agentCoords[1], OS.zomCoords[0], OS.zomCoords[1]);
    let zomAngle = utilsAI.angle([OS.agentCoords[0], OS.agentCoords[1]], [OS.zomCoords[0], OS.zomCoords[1]]); 
  
    let distancePenalty = zomDist; 
        
    let distancePenaltyScaling = observationSpace.xyNorm; // [0, 1] xyNorm is hardcoded 0.002 based on map size
    distancePenalty *= distancePenaltyScaling;
    let invDistPenalty = 1 - distancePenalty // [-1, 0]
    invDistPenalty /= 2; // [-0.5, 0] // share space
    
    penalty += invDistPenalty;
  
    function calculateTimeBonus(step, maxSteps) {
        const minBonus = 0.1; // step 0
        const maxBonus = 1;   // maxSteps
        const timeRatio = step / maxSteps;
        const timeBonus = minBonus + (maxBonus - minBonus) * timeRatio;
        return timeBonus;
    }
    //console.log(`Distance Penalty: ${distancePenalty}`);
    //console.log(`Inverted Distance Penalty: ${invDistPenalty}`);
    //console.log(`Angle Penalty: ${anglePenalty}`);
    //console.log(`total Reward: ${totalReward}`);
    OS.rawDist = zomDist;
    OS.rawAngle = zomAngle;
    osNS[0][LN.dist] = zomDist * OS.xyNorm;
    osNS[0][LN.angle] = (zomAngle + Math.PI) / (2 * Math.PI);
  
    const timeBonus = calculateTimeBonus(currentStep, agent.maxStepCount);
    reward += timeBonus;
    
    if (penalty > 1) {penalty = 1} // failsafe
    if (reward > 1) {reward = 1} // failsafe
    
    reward = reward - penalty;
   
    if (hitWall) { // huge penalty for walking into walls
      reward = -1; // walking into walls is bad
    } 
    if (zomDist <= player.width) {
      reward = -1; 
      console.log(`Zombie found user on step: ${currentStep}`);
      isDone = true;
    }
    else if (currentStep >= agent.maxStepCount) {
      reward = 1;
      isDone = true;
    }
    //console.log(`final reward: ${reward}`);
    
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
        UTC.state = envReset();
        UTC.currentStep = 0;
        UTC.newEpisode = false;
    }
    /*
    if (player.lastMoveX != 0 || player.lastMoveY != 0) {
        player.x += (player.lastMoveX * player.speed);
        player.y += (player.lastMoveY * player.speed);   
    }
    */
    let aX = player.lastMoveX;
    let aY = player.lastMoveY;
    
    const action = [aX, aY]
    const {next_state, reward, isDone} = simulateEnvStep(action, UTC.currentStep);

    for (let i = ents.length -1; i >= 0; i--) {
      if (!ents[i]) {continue}
      ents[i].draw();
  }

    //console.log(`next_state: ${next_state}`);
    agent.savexp(UTC.state, next_state, action, isDone, reward);

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

    observationSpace.initUpdate()
    UTC.currentStep = 0;
    UTC.currentEpisode = 0;
    UTC.newEpisode = true;
    UTC.animationTime = setInterval(animate, UTC.animateSpeed);
    //animate(); clearInterval(userTrainingControl.animationTime);
  }