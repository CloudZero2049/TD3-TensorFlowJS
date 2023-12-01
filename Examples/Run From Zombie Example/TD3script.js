const utilsAI = {
  distance: function(x1,y1,x2,y2) {
    return Math.floor(Math.hypot(x2 - x1, y2 - y1));
  },
  angle: function(from,to) { // [x,y], [x,y]
      let fX = from[0];
      let fY = from[1];
      let tX = to[0];
      let tY = to[1];
    
      let dx = tX - fX;
      let dy = tY - fY;
    
      let angle = Math.atan2(dy, dx);
   
      return angle;
  }
}

const observationSpace ={ 
  xyNorm: 0.002,
  initUpdate: function() {
    let defs = this.defaults;
  
    this.agentCoords[0] = parseInt(UI.aSliderX.value) + (player.width/2);
    this.agentCoords[1] = parseInt(UI.aSliderY.value) + (player.height/2);
    this.zomCoords[0] = parseInt(UI.zSliderX.value) + (zom1.width/2);
    this.zomCoords[1] = parseInt(UI.zSliderY.value) + (zom1.height/2);
    
    this.rawDist = utilsAI.distance(this.agentCoords[0], this.agentCoords[1], this.zomCoords[0], this.zomCoords[1]);
    this.rawAngle = utilsAI.angle([this.agentCoords[0],this.agentCoords[1]],[this.zomCoords[0], this.zomCoords[1]]);
    const os = this;
    defs[0][0] = this.agentCoords[0] * this.xyNorm;
    defs[0][1] = this.agentCoords[1] * this.xyNorm;
    defs[0][2] = this.zomCoords[0] * this.xyNorm;
    defs[0][3] = this.zomCoords[1] * this.xyNorm;
    defs[0][4] = (utilsAI.distance(os.agentCoords[0], os.agentCoords[1], os.zomCoords[0], os.zomCoords[1]) * os.xyNorm);
    defs[0][5] = (utilsAI.angle([os.agentCoords[0],os.agentCoords[1]],[os.zomCoords[0], os.zomCoords[1]])+ Math.PI) / (2 * Math.PI);

  },
  locNums: {
    aX: 0,
    aY: 1,
    zX: 2,
    zY: 3,
    dist: 4,
    angle: 5,
  },
  agentCoords: [player.x + (player.width/2), player.y + (player.height/2)],
  zomCoords: [zom1.x + (zom1.width/2), zom1.y + (zom1.height/2)],
  rawDist: utilsAI.distance(player.x + (player.width/2), player.y + (player.height/2), zom1.x + (zom1.width/2), zom1.y + (zom1.height/2)),
  rawAngle: utilsAI.angle([player.x + (player.width/2), player.y + (player.height/2)], [zom1.x + (zom1.width/2), zom1.y + (zom1.height/2)]),
  defaults: [[(player.x + (player.width/2)*this.xyNorm), (player.y + (player.height/2)*this.xyNorm), (zom1.x + (zom1.width/2)*this.xyNorm), (zom1.y + (zom1.height/2)*this.xyNorm), 1, 1]],
  stateSpace: [[(player.x + (player.width/2)*this.xyNorm), (player.y + (player.height/2)*this.xyNorm), (zom1.x + (zom1.width/2)*this.xyNorm), (zom1.y + (zom1.height/2)*this.xyNorm), 1, 1]],
  next_stateSpace: [[(player.x + (player.width/2)*this.xyNorm), (player.y + (player.height/2)*this.xyNorm), (zom1.x + (zom1.width/2)*this.xyNorm), (zom1.y + (zom1.height/2)*this.xyNorm), 1, 1]],
}
const actionSpace = {
  numberActions: 2,
  shape: [2], // not used
  actions: [0,0], // not used
  low: [-1,-1],
  high: [1,1]
};

class Actor {
  constructor(n_actions,inputShape) {
    this.model = tf.sequential();
    this.model.add(tf.layers.dense({ inputShape: inputShape, units: 512, activation: 'relu', dtype: 'float32', kernelInitializer: 'randomNormal'  }));
    this.model.add(tf.layers.dense({ units: 512, activation: 'relu', dtype: 'float32', kernelInitializer: 'randomNormal'   }));
    this.model.add(tf.layers.dense({ units: n_actions, activation: 'tanh', dtype: 'float32', kernelInitializer: 'randomNormal'  }));
    
  }
  
  call(stateTensor) {
    const pred = this.model.predict(stateTensor);
    return pred;
   
  }
}
class Agent {
  constructor(n_actions = 2, inputShapeA = 8) {                           
    this.actor_main = new Actor(n_actions,inputShapeA);
    
    this.loadedFiles = [];
    this.n_actions = n_actions;
    this.a_opt = tf.train.adam(0.001);
    this.maxStepCount = 256; 
    this.min_action = actionSpace.low[0];  
    this.max_action = actionSpace.high[0]; 

    this.actor_main.model.compile({optimizer: this.a_opt, loss: tf.losses.meanSquaredError}); 
    
  }

  act(state) { 
  
  let returnValue = tf.tidy(() => {     
   
  const stateTensor = tf.tensor(state);
  let actions = this.actor_main.call(stateTensor);
  const clippedActions = tf.clipByValue(actions, this.min_action, this.max_action); 
  const scaledActions = tf.mul(clippedActions, tf.scalar(this.max_action));
  return scaledActions.arraySync()[0]; 
});
return returnValue;
}
  
  async loadModels(modelFiles) {
    console.log(`Loading Files...`);
    
    try {
    for (let i=0; i < modelFiles.length; i++) {
      let name = modelFiles[i].name;
      switch(name){
        case "actor_main-model.json": {
            let weights;
            for (let j=0; j < modelFiles.length; j++) {
              if (modelFiles[j].name === "actor_main-model.weights.bin") {
                weights = modelFiles[j];
              }
            } // end j loop
            if (weights) {
              agent.actor_main.model = await tf.loadLayersModel(tf.io.browserFiles(
                [modelFiles[i], weights]));
                agent.actor_main.model.compile({optimizer: agent.a_opt, loss: tf.losses.meanSquaredError}); 
              console.log(`Loaded actor_main model`);
              const index = agent.loadedFiles.indexOf('actor_main');
              if (index !== -1) {agent.loadedFiles.splice(index, 1);}
              
              agent.loadedFiles.push(`actor_main`);
            }
            else {throw`actor_main weights file not found. Expecting: actor_main-model.weights.bin`}
          }
        break; 
      }
      UI.modelsLoadedInfo.innerHTML = `Models Loaded: ${agent.loadedFiles}`;
      UI.modelsLoadedInfo.style="color:rgb(10, 218, 10)";
      
    } // end i loop
    } catch (error) {
      console.error(`failed to load model: ${error}`);
    }
  }
} // End Agent Class

function envReset() {
  const os = observationSpace;
    os.agentCoords[0] = parseInt(UI.aSliderX.value) + (player.width/2);
    os.agentCoords[1] = parseInt(UI.aSliderY.value) + (player.height/2);
    player.x = parseInt(UI.aSliderX.value);
    player.y = parseInt(UI.aSliderY.value);
    os.zomCoords[0] = parseInt(UI.zSliderX.value) + (zom1.width/2);
    os.zomCoords[1] = parseInt(UI.zSliderY.value) + (zom1.height/2);
    zom1.x = parseInt(UI.zSliderX.value);
    zom1.y = parseInt(UI.zSliderY.value);

    os.rawDist = utilsAI.distance(os.agentCoords[0], os.agentCoords[1], os.zomCoords[0], os.zomCoords[1]);
    os.rawAngle = utilsAI.angle([os.agentCoords[0],os.agentCoords[1]],[os.zomCoords[0], os.zomCoords[1]]);

  if (UI.randZomCheckbox.checked) {
    let zomLoc = Game.getZombieSpawn(); // gives center
    os.zomCoords[0] = zomLoc[0];
    os.zomCoords[1] = zomLoc[1];
    os.defaults[0][os.locNums.zX] = (zomLoc[0] * os.xyNorm); // normalize
    os.defaults[0][os.locNums.zY] = (zomLoc[1] * os.xyNorm);

    zom1.x = zomLoc[0] - (zom1.width/2);
    zom1.y = zomLoc[1] - (zom1.height/2);
    
  }
  
  if (UI.randAgentCheckbox.checked) {
    let center = [os.zomCoords[0], os.zomCoords[1]];
    let agentLoc = Game.getAgentSpawn(center); // gives center
    os.agentCoords[0] = agentLoc[0];
    os.agentCoords[1] = agentLoc[1];
    os.defaults[0][os.locNums.aX] = (agentLoc[0] * os.xyNorm);
    os.defaults[0][os.locNums.aY] = (agentLoc[1] * os.xyNorm);

    player.x = agentLoc[0] - (player.width/2);
    player.y = agentLoc[1] - (player.height/2);
  }
 
  // Reset the state space
  os.stateSpace = JSON.parse(JSON.stringify(os.defaults));
  os.next_stateSpace = JSON.parse(JSON.stringify(os.defaults));
  
  const state = os.stateSpace;
  
  return state;
}



function envStep(action, currentStep) {
  const actionClone = JSON.parse(JSON.stringify(action));
  const OS = observationSpace;
  const osNS = observationSpace.next_stateSpace;
  const LN = observationSpace.locNums;
  let hitWall = false;
   const x = (player.speed * actionClone[0]); 
   const y = (player.speed * actionClone[1]);
  
  const movementThreshold = 0.15;
  // move X
  if ((actionClone[0] < -movementThreshold) || (actionClone[0] > movementThreshold)) {
    if ((OS.agentCoords[0] + x) < (player.width/2)) { // hit left wall
      OS.agentCoords[0] = (player.width/2); 
      osNS[0][LN.aX] = (player.width/2) * OS.xyNorm;
      hitWall = true;
    }
    else if ((OS.agentCoords[0] + x) > (Game.width - (player.width/2))) { // hit right wall
      OS.agentCoords[0] = (Game.width - (player.width/2));
      osNS[0][LN.aX] = (Game.width - (player.width/2)) * OS.xyNorm;
      hitWall = true;
    } 
    else {
      OS.agentCoords[0] += x;
      osNS[0][LN.aX] += x * OS.xyNorm;
    }
    
  }
  
  // move Y
  if ((actionClone[1] < -movementThreshold) || (actionClone[1] > movementThreshold)) {
    if ((OS.agentCoords[1] + y) < (player.height/2)) { // hit top wall
      OS.agentCoords[1] = (player.height/2);
      osNS[0][LN.aY] = (player.height/2) * OS.xyNorm;
      hitWall = true;
    }
    else if ((OS.agentCoords[1] + y) > (Game.height - (player.height/2))) { // hit bottom wall
      OS.agentCoords[1] = (Game.height - (player.height/2)); 
      osNS[0][LN.aY] = (Game.height - (player.height/2)) * OS.xyNorm; 
      hitWall = true;
    }
    else {
      OS.agentCoords[1] += y;
      osNS[0][LN.aY] += y * OS.xyNorm;
    }
   
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
let invDistPenalty = 1 - distancePenalty // [-1,0]
invDistPenalty /= 2; // [-0.5, 0] // share space
  
penalty += invDistPenalty;

function calculateTimeBonus(step, maxSteps) {
  const minBonus = 0.1; // step 0
  const maxBonus = 1;   // maxSteps
  const timeRatio = step / maxSteps;
  const timeBonus = minBonus + (maxBonus - minBonus) * timeRatio;
  return timeBonus;
}
  
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
  if (zomDist <= player.width) { // If zombie finds agent nothing else matters, game ends.
    reward = -1; 
    console.log(`Zombie found agent on step: ${currentStep}`);
    
    isDone = true;
  }
  else if (currentStep >= agent.maxStepCount) {
    reward = 1;
    console.warn("AGENT SURVIVED!"); // If the zombie hasn't reached the player then the agent survives, max point
    Game.agentWins++;
    UI.agentWins.innerHTML = `Times Won: ${Game.agentWins}`;
    isDone = true;
  }
  
  const next_State = JSON.parse(JSON.stringify(observationSpace.next_stateSpace));

  let terminalFlag = false;
  if (isDone) {terminalFlag = true}
  const aX = OS.agentCoords[0];
  const aY = OS.agentCoords[1];
  const zX = OS.zomCoords[0];
  const zY = OS.zomCoords[1];
  Game.agentMoves.push([aX, aY, terminalFlag]); // For drawing agent paths
  Game.zombieMoves.push([zX, zY]); // For drawing zombie paths
  return { next_state: next_State, reward: reward, isDone: isDone };
}

const agent = new Agent(actionSpace.numberActions, observationSpace.stateSpace[0].length, observationSpace.stateSpace[0].length + actionSpace.numberActions); 
let episodes = 5; // changes from slider
const epReward = [];
const totalAvgReward = [];

function main(epNum,stepSize,batchSize,warmupSteps) {
  console.log("Training Started");
  let target = false;
  if (epNum && !isNaN(epNum) && epNum > 0) {episodes = epNum}
  if (stepSize && !isNaN(stepSize) && stepSize > 0) {agent.maxStepCount = stepSize}
  
  observationSpace.initUpdate()

  for (let s = 0; s < episodes; s++) {
    if (!Game.running) {break}
    if (target) {
      break;
    }
  
    let totalReward = 0;
    let state = envReset();
    let done = false;
    let currentStep = 0;
    let stepDone = false;
   
    while (!done) {
      if (!Game.running) {break}
      const action = agent.act(state); 
      const { next_state, reward, isDone } = envStep(action, currentStep);
      
      state = JSON.parse(JSON.stringify(observationSpace.next_stateSpace)); // DEEP COPY
      
      totalReward += reward;
      
      if (currentStep >= agent.maxStepCount) {stepDone = true}
      currentStep++

      if (isDone || stepDone) {
        epReward.push(JSON.parse(JSON.stringify(totalReward)));
        const avgReward = epReward.slice(-100).reduce((a, b) => a + b, 0) / Math.min(epReward.length, 100);
        totalAvgReward.push(avgReward);

        console.log(`Total reward at Episode ${s} is ${totalReward} and avg reward is ${avgReward}`);

        if (Math.floor(avgReward) === 100) {
          target = true;
        }
        done = true;
        Game.episodesRan++;
        UI.episodesRan.innerHTML = `Episodes Ran: ${Game.episodesRan}`;
        
      }
    }
  }
  animateAgent();
}
