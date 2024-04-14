
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
  defaultsLength: 47,
  locNums: {
    aX: 0,
    aY: 1,
    s1: 2,
    s45:46, //rays
  },
}

const actionSpace = {
  numberActions: 2,
  shape: [0,2], // not used
  actions: [0,0], // not used
  low: [-1,-1],
  high: [1,1],
  
};

class Actor {
  constructor(n_actions,inputShape,nodes) { 
    this.model = tf.sequential();
    this.model.add(tf.layers.dense({ inputShape: inputShape, units: nodes, activation: 'relu', dtype: 'float32', kernelInitializer: 'heNormal',kernelRegularizer:tf.regularizers.l2({l2:0.1})}));
    this.model.add(tf.layers.dense({ units: nodes, activation: 'relu', dtype: 'float32', kernelInitializer: 'heNormal',kernelRegularizer:tf.regularizers.l2({l2:0.1})}));
    this.model.add(tf.layers.dense({ units: n_actions, activation: 'tanh', dtype: 'float32', kernelInitializer: 'glorotUniform',kernelRegularizer:tf.regularizers.l2({l2:0.1})}));
  }
  
  call(stateTensor, options = {}) { 
    const { batchSize = agent.batch_size } = options;
    const pred = this.model.predict(stateTensor, {batchSize});
    return pred;
  }
}

class Agent {
  constructor(n_actions = 2, inputShapeA = 47, runAlpha = 0.00001, chaseAlpha = 0.0001) {                           
    this.chaser_actor = new Actor(n_actions,inputShapeA,128);
    this.runner_actor = new Actor(n_actions,inputShapeA,522);
    this.loadedFiles = [];
    this.chaser_batch_size = 64;
    this.runner_batch_size = 256;
    this.n_actions = n_actions;
    this.runAlpha = runAlpha;
    this.chaseAlpha = chaseAlpha;
    this.run_opt = tf.train.adam(runAlpha);
    this.chase_opt = tf.train.adam(chaseAlpha);
    this.trainstep = 0;
    this.maxStepCount = 256;
    this.decScale = 100000;
    this.min_action = actionSpace.low[0];
    this.max_action = actionSpace.high[0];
    this.runner_actor.model.compile({optimizer: this.run_opt, loss: tf.losses.meanSquaredError});
    this.chaser_actor.model.compile({optimizer: this.chase_opt, loss: tf.losses.meanSquaredError});
  }

  act(state, actorModel) {
  let returnValue = tf.tidy(() => {
  const stateTensor = tf.tensor(state);
  let actions = actorModel.call(stateTensor, {batchSize: 1});
  const clippedActions = tf.clipByValue(actions, this.min_action, this.max_action); 
  const scaledActions = tf.mul(clippedActions, tf.scalar(this.max_action));
  return scaledActions.arraySync()[0];
});
return returnValue;
}
  
async loadChaserModel(modelFiles) {
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
            agent.chaser_actor.model = await tf.loadLayersModel(tf.io.browserFiles(
              [modelFiles[i], weights]));
            agent.chaser_actor.model.compile({optimizer: agent.chase_opt, loss: tf.losses.meanSquaredError}); 
            console.log(`Loaded chaser model`);
            const index = agent.loadedFiles.indexOf('chaser_actor');
            if (index !== -1) {agent.loadedFiles.splice(index, 1);}
            agent.loadedFiles.push(`chaser_actor`);
          }
          else {throw`chaser weights file not found. Expecting: actor_main-model.weights.bin`}
        }
      break;
          
    }
    if (agent.loadedFiles.length > 0){
      let modelString = `Models Loaded:`;
      for (let file of agent.loadedFiles) {
        let index = agent.loadedFiles.indexOf(file);
        modelString = modelString + ` ` + `[${index + 1}]` + file + `,`;
      }
      UI.modelsLoadedInfo.innerHTML = modelString;
      UI.modelsLoadedInfo.style="color:rgb(10, 218, 10)";
  }
  } // end i loop
  } catch (error) {
    console.error(`failed to load model: ${error}`);
  }
} // end load chaser model

async loadRunnerModel(modelFiles) {
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
            agent.runner_actor.model = await tf.loadLayersModel(tf.io.browserFiles(
              [modelFiles[i], weights]));
            agent.runner_actor.model.compile({optimizer: agent.run_opt, loss: tf.losses.meanSquaredError}); 
            console.log(`Loaded runner model`);
            const index = agent.loadedFiles.indexOf('runner_actor');
            if (index !== -1) {agent.loadedFiles.splice(index, 1);}
            
            agent.loadedFiles.push(`runner_actor`);
            
          }
          else {throw`runner weights file not found. Expecting: actor_main-model.weights.bin`}
        }
      break;  
    }
    if (agent.loadedFiles.length > 0){
      let modelString = `Models Loaded:`;
      for (let file of agent.loadedFiles) {
        let index = agent.loadedFiles.indexOf(file);
        modelString = modelString + ` ` + `[${index + 1}]` + file + `,`;
      }
      UI.modelsLoadedInfo.innerHTML = modelString;
      UI.modelsLoadedInfo.style="color:rgb(10, 218, 10)";
  }
  } // end i loop
  } catch (error) {
    console.error(`failed to load model: ${error}`);
  }
} // end load runner model
 
} // End Agent Class

function envReset() {
  const os = observationSpace;
  const ents = Game.entities;
  let tempEnts = [];
  for (let i = 0; i < ents.length; i++) {
    let spawn = Game.getSpawn(tempEnts); // [x,y] gives center
    ents[i].x = spawn[0] - 10;
    ents[i].y = spawn[1] - 10;
    ents[i].location[0] = spawn[0];
    ents[i].location[1] = spawn[1];
    let baseX = ents[i].location[0] * os.xyNorm;
    let baseY = ents[i].location[1] * os.xyNorm;
    let floorX = Math.floor(baseX * agent.decScale);
    let floorY = Math.floor(baseY * agent.decScale);
    ents[i].defaults[0][os.locNums.aX] = (floorX / agent.decScale);
    ents[i].defaults[0][os.locNums.aY] = (floorY / agent.decScale);
    tempEnts.push(ents[i]);
  }
  for (let i = 0; i < Game.chaserNum; i++) {
    ents[i].role = "chaser";
    ents[i].color = Game.getColor("chaser");
    ents[i].speed = tagControl.chaserSpeed;    
  }
  for (let i = Game.chaserNum; i < ents.length; i++) {
    ents[i].role = "runner";
    ents[i].color = Game.getColor("runner");
    ents[i].speed = tagControl.runnerSpeed;
  }
  for (let i = ents.length -1; i >= 0; i--) {
    if (!ents[i]) {continue}
    ents[i].createPolygon();
  } 
  for (let i = ents.length -1; i >= 0; i--) { // sensor data added to defaults here
    if (!ents[i]) {continue}
    ents[i].role == "chaser" ?  ents[i].updateChaserSensor(true) : ents[i].updateRunnerSensor(true);
  }
  for (let i = ents.length -1; i >= 0; i--) { // sensor data added to defaults here
    if (!ents[i]) {continue}
    ents[i].stateSpace = JSON.parse(JSON.stringify(ents[i].defaults));
    ents[i].nextStateSpace = JSON.parse(JSON.stringify(ents[i].defaults));
  }
}

const agent = new Agent(actionSpace.numberActions, observationSpace.defaultsLength, observationSpace.defaultsLength + actionSpace.numberActions); 
