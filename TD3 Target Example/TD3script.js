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
  //[agentX,agentY, directionX, directionY, agentSpeedX ,agentSpeedY,civilianX,civilianY,dist to civ, angle to civ] 10 total
  initUpdate: function() { // [0,0],[0,0]
    let defs = this.defaults;
  
    this.agentCoords[0] = parseInt(UI.aSliderX.value) + (player.width/2);
    this.agentCoords[1] = parseInt(UI.aSliderY.value) + (player.height/2);
    this.civCoords[0] = parseInt(UI.cSliderX.value) + (civ1.width/2);
    this.civCoords[1] = parseInt(UI.cSliderY.value) + (civ1.height/2);

    this.rawDist = utilsAI.distance(this.agentCoords[0], this.agentCoords[1], this.civCoords[0], this.civCoords[1]);
    this.rawAngle = utilsAI.angle([this.agentCoords[0],this.agentCoords[1]],[this.civCoords[0], this.civCoords[1]]);
    const os = this;
    defs[0][0] = this.agentCoords[0] * this.xyNorm; // normalizing values
    defs[0][1] = this.agentCoords[1] * this.xyNorm;
    defs[0][2] = this.civCoords[0] * this.xyNorm;
    defs[0][3] = this.civCoords[1] * this.xyNorm;
    defs[0][4] = (utilsAI.distance(os.agentCoords[0], os.agentCoords[1], os.civCoords[0], os.civCoords[1]) * os.xyNorm);
    defs[0][5] = (utilsAI.angle([os.agentCoords[0],os.agentCoords[1]],[os.civCoords[0], os.civCoords[1]])+ Math.PI) / (2 * Math.PI);

  },
  locNums: {
    aX: 0,
    aY: 1,
    cX: 2,
    cY: 3,
    dist: 4,
    angle: 5,
  },
  agentCoords: [player.x + (player.width/2), player.y + (player.height/2)],
  civCoords: [civ1.x + (civ1.width/2), civ1.y + (civ1.height/2)],
  rawDist: utilsAI.distance(player.x + (player.width/2), player.y + (player.height/2), civ1.x + (civ1.width/2), civ1.y + (civ1.height/2)),
  rawAngle: utilsAI.angle([player.x + (player.width/2), player.y + (player.height/2)], [civ1.x + (civ1.width/2), civ1.y + (civ1.height/2)]),
  defaults: [[(player.x + (player.width/2)*this.xyNorm), (player.y + (player.height/2)*this.xyNorm), (civ1.x + (civ1.width/2)*this.xyNorm), (civ1.y + (civ1.height/2)*this.xyNorm), 1, 1]],
  stateSpace: [[(player.x + (player.width/2)*this.xyNorm), (player.y + (player.height/2)*this.xyNorm), (civ1.x + (civ1.width/2)*this.xyNorm), (civ1.y + (civ1.height/2)*this.xyNorm), 1, 1]],
  next_stateSpace: [[(player.x + (player.width/2)*this.xyNorm), (player.y + (player.height/2)*this.xyNorm), (civ1.x + (civ1.width/2)*this.xyNorm), (civ1.y + (civ1.height/2)*this.xyNorm), 1, 1]],
}
// state: [agentX, agentY, civX, civY, distance, angle]
// actions: [directionX, directionY, speedX, speedY]
const actionSpace = {
  numberActions: 4,
  shape: [4], // not used
  actions: [0,0,0,0], // not used
  low: [-1,-1,-1,-1],
  high: [1,1,1,1]
};

class Actor {
  constructor(n_actions,inputShape) { // has alpha in constructor (video)
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
  constructor(n_actions = 4, inputShapeA = 6) {                           
    this.actor_main = new Actor(n_actions,inputShapeA);
    this.loadedFiles = [];
    this.n_actions = n_actions;
    this.a_opt = tf.train.adam(0.001);
    this.maxStepCount = 64;
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
    os.civCoords[0] = parseInt(UI.cSliderX.value) + (civ1.width/2);
    os.civCoords[1] = parseInt(UI.cSliderY.value) + (civ1.height/2);

    player.x = parseInt(UI.aSliderX.value);
    player.y = parseInt(UI.aSliderY.value);
    civ1.x = parseInt(UI.cSliderX.value);
    civ1.y = parseInt(UI.cSliderY.value);

    os.rawDist = utilsAI.distance(os.agentCoords[0], os.agentCoords[1], os.civCoords[0], os.civCoords[1]);
    os.rawAngle = utilsAI.angle([os.agentCoords[0],os.agentCoords[1]],[os.civCoords[0], os.civCoords[1]]);

  if (UI.randCivCheckbox.checked) {
    let civLoc = Game.getCivilianSpawn();
    os.civCoords[0] = civLoc[0];
    os.civCoords[1] = civLoc[1];
    os.defaults[0][os.locNums.cX] = (civLoc[0] * os.xyNorm); // normalize
    os.defaults[0][os.locNums.cY] = (civLoc[1] * os.xyNorm);

    //Push civ coords for drawing
    Game.civilianMoves.push([civLoc[0], civLoc[1]]);
  }
  
  if (UI.randAgentCheckbox.checked) {
    let center = [os.civCoords[0], os.civCoords[1]];
    let agentLoc = Game.getAgentSpawn(center);
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
  const LN = observationSpace.locNums
  const stateCoords = JSON.parse(JSON.stringify([OS.agentCoords[0], OS.agentCoords[1]])); 
  let hitWall = false;
  
   const x = (player.speed * Math.abs(actionClone[2])); 
   const y = (player.speed * Math.abs(actionClone[3]));
   
  // move X
  const movementThreshold = 0.15;

  if (actionClone[0] < -movementThreshold) { // move left
    if ((OS.agentCoords[0] - x) < (player.width/2)) {
      OS.agentCoords[0] = (player.width/2); 
      osNS[0][LN.aX] = (player.width/2) * OS.xyNorm;
      hitWall = true;
    }
    else {
      OS.agentCoords[0] -= x;
      osNS[0][LN.aX] -= x * OS.xyNorm;
    }
    
  }
  else if (actionClone[0] > movementThreshold) { // move right
    if ((OS.agentCoords[0] + x) > (Game.width - (player.width/2))) {
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
  if (actionClone[1] < -movementThreshold) { // move up
    if ((OS.agentCoords[1] - y) < (player.height/2)) {
      OS.agentCoords[1] = (player.height/2);
      osNS[0][LN.aY] = (player.height/2) * OS.xyNorm;
      hitWall = true;
    }
    else {
      OS.agentCoords[1] -= y;
      osNS[0][LN.aY] -= y * OS.xyNorm;
    }
  }
  else if (actionClone[1] > movementThreshold) { // move down
    if ((OS.agentCoords[1] + y) > (Game.height - (player.height/2))) {
      OS.agentCoords[1] = (Game.height - (player.height/2)); 
      osNS[0][LN.aY] = (Game.height - (player.height/2)) * OS.xyNorm; 
      hitWall = true;
    }
    else {
      OS.agentCoords[1] += y
      osNS[0][LN.aY] += y * OS.xyNorm;
    }
  }
 
  let isDone = false;
  let reward = 0;
  let penalty = 0;
  let civDist = utilsAI.distance(OS.agentCoords[0], OS.agentCoords[1], OS.civCoords[0], OS.civCoords[1]);
  let civAngle = utilsAI.angle([OS.agentCoords[0], OS.agentCoords[1]], [OS.civCoords[0], OS.civCoords[1]]);
  let agentHeading = utilsAI.angle([stateCoords[0], stateCoords[1]], [OS.agentCoords[0], OS.agentCoords[1]]); 

  function angleDifferenceRadians(angle1, angle2) {
    let diff = Math.abs(angle1 - angle2);
    return Math.min(diff, 2 * Math.PI - diff);
}

let angleRadDif = angleDifferenceRadians(civAngle, agentHeading);

let anglePenalty = angleRadDif / Math.PI; 
anglePenalty /= 2;

let distancePenalty = civDist; 

let distancePenaltyScaling = observationSpace.xyNorm;
distancePenalty *= distancePenaltyScaling;
distancePenalty /= 2;

penalty += anglePenalty + distancePenalty;
 
function calculateTimePenalty(step, maxSteps) {
  const maxPenalty = -0.1; // Adjust as needed
  const timeRatio = step / maxSteps;
  const timePenalty = Math.max(maxPenalty, maxPenalty * timeRatio);
  return timePenalty;
}
 
OS.rawDist = civDist;
OS.rawAngle = civAngle;
osNS[0][LN.dist] = civDist * OS.xyNorm;
osNS[0][LN.angle] = (civAngle + Math.PI) / (2 * Math.PI);

const timePenalty = calculateTimePenalty(currentStep, agent.maxStepCount);

  penalty += Math.abs(timePenalty);
  if (hitWall) {
    penalty += 0.5;
  } 
  if (penalty > 1) {penalty = 1} // failsafe
  
  reward = reward - penalty;
  
  
  if (civDist <= player.width) {
    reward = 1; 
    console.warn("AGENT FOUND CIVILIAN!");
    Game.agentWins++;
    UI.agentWins.innerHTML = `Times Won: ${Game.agentWins}`;
    isDone = true;
  }
  
  const next_State = JSON.parse(JSON.stringify(observationSpace.next_stateSpace));
  
  if (currentStep >= agent.maxStepCount) {isDone = true}
  let terminalFlag = false;
  if (isDone) {terminalFlag = true}
  const aX = OS.agentCoords[0];
  const aY = OS.agentCoords[1];
  Game.agentMoves.push([aX, aY, terminalFlag]);
  return { next_state: next_State, reward: reward, isDone: isDone };
}

const agent = new Agent(actionSpace.numberActions, observationSpace.stateSpace[0].length, observationSpace.stateSpace[0].length + actionSpace.numberActions); 
let episodes = 5;
const epReward = [];
const totalAvgReward = [];

function main(epNum,stepSize) {
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
     
      state = JSON.parse(JSON.stringify(observationSpace.next_stateSpace));
      
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
