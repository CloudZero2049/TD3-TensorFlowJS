
const utilsAI = {
  criticLoss1: 0,
  criticLoss2: 0,
  actorLoss: 0,
  doneFlag: false,
  logLosses: function() { // logs the most recent loss values at end of all episodes
    console.table({critic1:utilsAI.criticLoss1,critic2:utilsAI.criticLoss2,actor:utilsAI.actorLoss});
  },
  logMemory:function() {
    console.log(`Memory Used: ${agent.memory.cnt}. Max Memory: ${agent.memory.maxsize}`);
  },
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
    //this.civCoords[0] = parseInt(UI.cSliderX.value) + (civ1.width/2);
    //this.civCoords[1] = parseInt(UI.cSliderY.value) + (civ1.height/2);
    this.zomCoords[0] = parseInt(UI.zSliderX.value) + (zom1.width/2);
    this.zomCoords[1] = parseInt(UI.zSliderY.value) + (zom1.height/2);
    
    player.x = parseInt(UI.aSliderX.value);
    player.y = parseInt(UI.aSliderY.value);
    zom1.x = parseInt(UI.zSliderX.value);
    zom1.y = parseInt(UI.zSliderY.value);
    //civ1.x = parseInt(UI.cSliderX.value);
    //civ1.y = parseInt(UI.cSliderY.value);

    //this.rawDist = utilsAI.distance(this.agentCoords[0], this.agentCoords[1], this.zomCoords[0], this.zomCoords[1]);
    //this.rawAngle = utilsAI.angle([this.agentCoords[0],this.agentCoords[1]],[this.zomCoords[0], this.zomCoords[1]]);
    const os = this;
    //defs[0][0] = parseFloat((this.agentCoords[0] * this.xyNorm).toFixed(4)); // normalizing values
    //defs[0][1] = parseFloat((this.agentCoords[1] * this.xyNorm).toFixed(4));
    let aX = this.agentCoords[0] * os.xyNorm;
    let aY = this.agentCoords[1] * os.xyNorm;
    defs[0][0] = (Math.floor((aX)*100000) / 100000);
    defs[0][1] = (Math.floor((aY)*100000) / 100000);
    //defs[0][2] = parseFloat((this.zomCoords[0] * this.xyNorm).toFixed(8));
    //defs[0][3] = parseFloat((this.zomCoords[1] * this.xyNorm).toFixed(8));
    //defs[0][27] = parseFloat(((utilsAI.distance(os.agentCoords[0], os.agentCoords[1], os.zomCoords[0], os.zomCoords[1]) * os.xyNorm)).toFixed(3));
    //defs[0][5] = parseFloat(((utilsAI.angle([os.agentCoords[0],os.agentCoords[1]],[os.zomCoords[0], os.zomCoords[1]])+ Math.PI) / (2 * Math.PI)).toFixed(8));
    
   const ents = Game.entities;
    for (let i = ents.length -1; i >= 0; i--) {
      if (!ents[i] || ents[i].type == "player") {continue}
      ents[i].createPolygon();
    }
    
    player.sensor.update(Game.mapBorders,Game.entities); // hardcoded zombie
    const offsetsXY = player.sensor.readings.map((reading) => {
      if (reading == null) {
          return null;  // Default value for no detection
      } else { 
          // Adjust the offset based on the type of the detected object
          if (reading.ent === "zombie") {
              
              return {xy:[reading.x,reading.y],type:"zombie"};

          } else if (reading.ent === "wall") { 

              return {xy:[reading.x,reading.y],type:"wall"};
              
          } else {
              // Default value for unknown types
              return null;
          }
      }
  });

 
  for (let i = 0; i < offsetsXY.length; i++){
    let distance;
    
    if (offsetsXY[i] == null) {distance = 1}
    else {
      distance = utilsAI.distance(os.agentCoords[0], os.agentCoords[1], offsetsXY[i].xy[0], offsetsXY[i].xy[1]);
      distance *= os.xyNorm; //scaled (0,1)
      if (offsetsXY[i].type == "zombie") {distance = -(1 - distance);} // (-1,0) descending for zombie
      else {distance = 1 - distance} //(0,1) ascending for walls
      
      distance = Math.floor(distance * agent.decScale);
      distance /=  agent.decScale;
      
      
    }
    
    defs[0][i+2] = distance;
  }
  },
  locNums: {
    aX: 0,
    aY: 1,
    s1: 2,
    s45:46,
    
  },
  agentCoords: [player.x + (player.width/2), player.y + (player.height/2)],
  //civCoords: [civ1.x + (civ1.width/2), civ1.y + (civ1.height/2)],
  zomCoords: [zom1.x + (zom1.width/2), zom1.y + (zom1.height/2)],
  rawDist: utilsAI.distance(player.x + (player.width/2), player.y + (player.height/2), zom1.x + (zom1.width/2), zom1.y + (zom1.height/2)),
  //rawAngle: utilsAI.angle([player.x + (player.width/2), player.y + (player.height/2)], [civ1.x + (civ1.width/2), civ1.y + (civ1.height/2)]),
  rawAngleDiff: 0,
  defaults: [[parseFloat((player.x + (player.width/2)*this.xyNorm).toFixed(5)), parseFloat((player.y + (player.height/2)*this.xyNorm).toFixed(5)),
   1,1,1,1,1,1,1,1,1,1,
   1,1,1,1,1,1,1,1,1,1,
   1,1,1,1,1,1,1,1,1,1,
   1,1,1,1,1,1,1,1,1,1,
   1,1,1,1,1
]],
  stateSpace: [[]],
  next_stateSpace: [[]],
}
// state: [agentX, agentY, rays 1-25, distance]
// actions: [move X, move Y]
const actionSpace = {
  numberActions: 2,
  shape: [0,2], // not used
  actions: [0,0], // not used
  low: [-1,-1],
  high: [1,1],
  randAct: function() {

  }
};

class RBuffer {
  constructor(maxsize, statedim, n_actions) { // number of components of an action
    this.cnt = 0;
    this.maxsize = maxsize; // memSize - maxsize
    this.state_memory = tf.zeros([maxsize, ...statedim]); // maxsize, *inputshape
    this.next_state_memory = tf.zeros([maxsize, ...statedim]);
    this.action_memory = tf.zeros([maxsize, n_actions]);
    this.reward_memory = tf.zeros([maxsize]);
    this.done_memory = tf.zeros([maxsize]); // "terminal_memory of "done" flags"
  }
  
  storexp(state, next_state, action, done, reward) { // self, state, action, reward, state_, done
    const index = this.cnt % this.maxsize;
    done = done ? 1 : 0; // force numbers

    this.state_memory[index] = JSON.parse(JSON.stringify(state));
    this.next_state_memory[index] = JSON.parse(JSON.stringify(next_state));
    this.action_memory[index] = JSON.parse(JSON.stringify([action])); // [] because action is [], math expects [[]]
    this.reward_memory[index] = JSON.parse(JSON.stringify(reward));
    this.done_memory[index] = JSON.parse(JSON.stringify(1-done)); // 1-done here because array math later. Number(done). = 0 is done
    this.cnt += 1;
    
  }

  sample(batch_size) {
    try {
    if (this.maxsize < batch_size) {
      throw`Agent's RBufferSize for memory must be >= batch_size. RBufferSize: ${this.maxsize}, batch_size: ${batch_size}`;
    }
    const max_mem = Math.min(this.cnt, (this.maxsize-1));

    // Randomly sample indices without replacement
    let batch = [];
    for (let i = 0; i < batch_size; i++) {
        let index;
        do {
            index = Math.floor(Math.random() * max_mem); 
        } while (batch.includes(index));
        batch.push(index);
    }
    //console.log(batch);
    const states = batch.map(index => [...this.state_memory[index]]);
    const next_states = batch.map(index => [...this.next_state_memory[index]]);
    //const rewards = batch.map(index => this.reward_memory[index]);
    const rewards = batch.map(index => JSON.parse(JSON.stringify(this.reward_memory[index])));
    const actions = batch.map(index => [...this.action_memory[index]]);
    const dones = batch.map(index => JSON.parse(JSON.stringify(this.done_memory[index])));
    
    return [states, next_states, rewards, actions, dones];
  }
  catch (error) {
    console.error(`failed to sample from memory: ${error}`);
  }
  }
}

class Critic { 
  constructor(inputShape) {
    this.model = tf.sequential(); //  original paper is 400,300, gitHub has 256,256, claude 3: 512 to 1024 (mid 768)
    this.model.add(tf.layers.dense({ inputShape: [inputShape], units: 576, activation: 'relu', dtype: 'float32', kernelInitializer: 'heNormal',kernelRegularizer:tf.regularizers.l2({l2:0.1})}));
    this.model.add(tf.layers.dense({ units: 576, activation: 'relu', dtype: 'float32', kernelInitializer: 'heNormal',kernelRegularizer:tf.regularizers.l2({l2:0.1}) }));
    this.model.add(tf.layers.dense({ units: 1, activation: null, dtype: 'float32',kernelInitializer: 'heNormal', kernelRegularizer:tf.regularizers.l2({l2:0.1}) })); // Output to 1 Q value
    
  }

  call(inputstate, actions) { // feed forward
   
   // STEP 10b: we concatinate the input state with actio
  // Concatenate the two tensors along the last axis (axis 1)
    const catTensor = tf.concat([inputstate, actions], 1);
   
    const pred = this.model.predict(catTensor, {batchSize: agent.batch_size});
    
    return pred;
  }
}
//kernelInitializer: tf.heNormal(shape)
class Actor {
  constructor(n_actions,inputShape) { // claude 3: 256 or 512 
    this.model = tf.sequential();
    this.model.add(tf.layers.dense({ inputShape: inputShape, units: 522, activation: 'relu', dtype: 'float32', kernelInitializer: 'heNormal',kernelRegularizer:tf.regularizers.l2({l2:0.1})  }));
    this.model.add(tf.layers.dense({ units: 522, activation: 'relu', dtype: 'float32', kernelInitializer: 'heNormal',kernelRegularizer:tf.regularizers.l2({l2:0.1})   }));
    this.model.add(tf.layers.dense({ units: n_actions, activation: 'tanh', dtype: 'float32', kernelInitializer: 'glorotUniform',kernelRegularizer:tf.regularizers.l2({l2:0.1})  })); // output units: is number of actions/action space
    // from DDPG paper. pi - tangent hyperbolic, +-1
    // if action bounds are say +-2, multiply that by tanh function before predict/output
  }
  
  call(stateTensor, options = {}) {
    const { batchSize = agent.batch_size } = options;
    
    // , {batchSize: 4} defaults to 32
    
    const pred = this.model.predict(stateTensor, {batchSize});
    
    return pred;
   
  }
  randAct() {
    const randX = (Math.random() * 2 - 1).toFixed(7); //  random num between -1 and 1
    const randY = (Math.random() * 2 - 1).toFixed(7); 

    return [[parseFloat(randX), parseFloat(randY)]];
  }
}
//alpha,beta,input_dims,tau,env,gamma,update_actor_interval = 2, warmup = 1000, n_actions=2,max_size=1000000,layer1_size=400,layer2_size=300, batch_size=100,noise=0.1 (video)
//min-max actions is because of noise (video)
// alpha = learning rate for actor (.001), beta = learning rate for critic (.002), tau = target weight update rate (slow is good)
//gamma = discount factor (0 is immediate rewards, 1 is long term rewards and possibly more exploration)
//Phil: batch_size = 300, warmup = 1000, n_games 1000. temp: [alpha = 0.00004, beta = 0.00005,]
//n_actions = 2, cInputShape = 9, alpha = 0.001, beta = 0.002, gamma = 0.99, tau = 0.005, warmup = 50, RBufferSize = 1000000 (300000)
class Agent {
  constructor(n_actions = 2, inputShapeA = 47, inputShapeC = 49, alpha = 0.00001, beta = 0.00007, gamma = 0.999, tau = 0.005, warmup = 225, RBufferSize = 500000) {                           
    this.actor_main = new Actor(n_actions,inputShapeA);
    this.actor_target = new Actor(n_actions,inputShapeA);
    this.critic_main = new Critic(inputShapeC);
    this.critic_main2 = new Critic(inputShapeC);
    this.critic_target = new Critic(inputShapeC);
    this.critic_target2 = new Critic(inputShapeC);
    this.loadedFiles = [];
    this.batch_size = 64; // Original paper uses 100 from entire agent history
    this.n_actions = n_actions;
    this.alpha = alpha;
    this.beta = beta;
    this.a_opt = tf.train.adam(alpha);
    this.c_opt1 = tf.train.adam(beta);
    this.c_opt2 = tf.train.adam(beta);
    this.memory = new RBuffer(RBufferSize, [observationSpace.defaults[0].length], this.n_actions); // [stateSpace dimentions]
    this.gamma = gamma;
    this.tau = tau;
    this.actor_update_steps = 2; // actor and targets update slower than critic in TD3
    this.warmup = warmup;
    this.trainstep = 0;
    this.maxStepCount = 256; // not tied to trainstep
    this.decScale = 100000;
    this.min_action = actionSpace.low[0];   // negative movement
    this.max_action = actionSpace.high[0];  // positive movement

    this.actor_main.model.compile({optimizer: this.a_opt, loss: tf.losses.meanSquaredError}); 
    this.actor_target.model.compile({optimizer: this.a_opt, loss: tf.losses.meanSquaredError}); 
    this.critic_main.model.compile({optimizer: this.c_opt1, loss: tf.losses.meanSquaredError});
    this.critic_target.model.compile({optimizer: this.c_opt1, loss: tf.losses.meanSquaredError});
    this.critic_main2.model.compile({optimizer: this.c_opt2, loss: tf.losses.meanSquaredError}); 
    this.critic_target2.model.compile({optimizer: this.c_opt2, loss: tf.losses.meanSquaredError}); 
    
    this.updateTargets(1); // tau = 1 for first update to cause a hard update. target networks gets set to main networks
  }

  act(state, activeTraining = false) {
  if (this.memory.cnt > this.warmup) {
    activeTraining = true;
  }
  let returnValue = tf.tidy(() => {     
  
  const stateTensor = tf.tensor(state);
 
  // STEP 1b: get actions from actor_main
  let actions;
  /*
  const noiseStep = 0.0000005
  const startNoise = 0.1;
  const minNoise = 0.01
  let baseNoise = startNoise - (agent.memory.cnt * noiseStep); // -0.01125 every 22,500 steps (100 episodes at 225 steps)
  const finalNoise = Math.max(minNoise, baseNoise);
    */
 
  if (!activeTraining && !Game.testing) { // warmup with random actions
    const randActions = this.actor_main.randAct();
   
    actions = tf.tensor(randActions);
    
    const noise = tf.randomNormal(actions.shape, 0.0, 0.1); // exploration noise, orig paper says 0.1
    
    actions = tf.add(actions, noise);
  }
  else if(!Game.testing){
    actions = this.actor_main.call(stateTensor, {batchSize: 1});

    const noise = tf.randomNormal(actions.shape, 0.0, 0.1); 
   
    actions = tf.add(actions, noise);
  }
  else { // testing active, deployment for real values.
    actions = this.actor_main.call(stateTensor, {batchSize: 1});
  }
  

  if (UI.trainModeCheckbox.checked) {
    const noise = tf.randomNormal(actions.shape, 0.0, 0.2); 
    actions = tf.add(actions, noise);
  }
  
  // clip actions because noise can cause them to go outside the bounds
  const clippedActions = tf.clipByValue(actions, this.min_action, this.max_action); 
  const scaledActions = tf.mul(clippedActions, tf.scalar(this.max_action));
 
  // Step 1c: returned action is the first element of a flatened clipped action array
  return scaledActions.arraySync()[0];  //--> [1,-1]
}); // end tidy
return returnValue;
}
  
  savexp(state, next_state, action, done, reward) { 
    this.memory.storexp(state, next_state, action, done, reward);
  }

  updateTargets(tau = null) { 
    if (tau == null) {
      tau = this.tau;
    }
    
    // STEP 14: we update the weights of the Actor target by polyak averaging
    const weights1 = [];
  
    const targets1 = this.actor_target.model.getWeights().map(w => w.clone());
    const mainWeights1 = this.actor_main.model.getWeights().map(w => w.clone());
    
    for (let i = 0; i < mainWeights1.length; i++) {
      const updatedWeight1 = tf.tidy(() => {
        const weightedMain1 = tf.mul(mainWeights1[i], tf.scalar(tau));
       
        const weightedTarget1 = tf.mul(targets1[i], tf.scalar(1 - tau)); // scalar tau only?
       
        const combinedWeight1 = tf.add(weightedMain1, weightedTarget1);
        
        return combinedWeight1;
      }); // end tidy
      //console.log(`updatedWeight ${updatedWeight}`);
      weights1.push(updatedWeight1);
      
      
    } ;// end loop 1
    
    this.actor_target.model.setWeights(weights1);
    
    // STEP 15: we update the weights of the Critic targets by polyak averaging
    const weights2 = [];
    const targets2 = this.critic_target.model.getWeights().map(w => w.clone());
    const mainWeights2 = this.critic_main.model.getWeights().map(w => w.clone());
    
    for (let i = 0; i < mainWeights2.length; i++) {
      const updatedWeight2 = tf.tidy(() => {
        const weightedMain2 = tf.mul(mainWeights2[i], tf.scalar(tau));
        const weightedTarget2 = tf.mul(targets2[i], tf.scalar(1 - tau));
        const combinedWeight2 = tf.add(weightedMain2, weightedTarget2);
        return combinedWeight2;
      }); // end tidy
      weights2.push(updatedWeight2);
     
    } // end loop

    this.critic_target.model.setWeights(weights2);
   
    const weights3 = [];
    const targets3 = this.critic_target2.model.getWeights().map(w => w.clone());
    const mainWeights3 = this.critic_main2.model.getWeights().map(w => w.clone());
    
    for (let i = 0; i < mainWeights3.length; i++) {
      const updatedWeight3 = tf.tidy(() => {
        const weightedMain3 = tf.mul(mainWeights3[i], tf.scalar(tau));
        const weightedTarget3 = tf.mul(targets3[i], tf.scalar(1 - tau));
        const combinedWeight3 = tf.add(weightedMain3, weightedTarget3);
        return combinedWeight3;
      }); // end tidy
      weights3.push(updatedWeight3);
      
    } // end loop

    this.critic_target2.model.setWeights(weights3);
    //video does some weird dictionary stuff here
  }
  // STEPS 4-15...
  train() {
    
    if (agent.memory.cnt < agent.batch_size) { // important, sample will fail othewise
        return; 
    }
    tf.tidy(() => {
    // STEP 4: we sample a batch of transitions (s, s`, a, r) from memory
    
    const [states, nextStates, rewards, actions, dones] = this.memory.sample(agent.batch_size);
    
    // video has all these using critic_1, says doesnt matter. dtype matters
    const statesTensor = tf.tensor(states).squeeze();
    const nextStatesTensor = tf.tensor(nextStates).squeeze();
    const rewardsTensor = tf.tensor(rewards); //[]
    const actionsTensor = tf.tensor(actions).squeeze(); 
    const donesTensor = tf.tensor(dones) // []
    
    // video zeros the optimizers here?

      // STEP 10a: the two critic models take each the couple (s, a) as input and return Q-values as outputs: Q1(s, a), Q2(s, a)
      // STEP 11: we compute the loss coming from the two Critic models: criticLoss = MSE_Loss(Q1(s, a), Qt) + MSE_Loss(Q2(s, a), Qt)

    function lossFunction1() { 
      const targetActions1 = agent.actor_target.call(nextStatesTensor, {batchSize: agent.batch_size});
      const nextStates1 = tf.clone(nextStatesTensor); 
      const rewards1 = tf.clone(rewardsTensor);
      const dones1 = tf.clone(donesTensor);
      const states1 = tf.clone(statesTensor);
      const actions1 = tf.clone(actionsTensor);

      // STEP 6: We add Gaussian noise to the next action a` and we clamp it in a range of values supported by environment
      const ranNormalActions1 = tf.add(targetActions1, tf.clipByValue(tf.randomNormal(targetActions1.shape, 0.0, 0.2), -0.5, 0.5));//.map(tf.clone); // video 2 uses actionsTensor?
      
      const clipActions1 = tf.mul(tf.scalar(agent.max_action), tf.clipByValue(ranNormalActions1, agent.min_action, agent.max_action));//.map(tf.clone); 
      
      // STEP 7: The two Critic targets take each the couple (s`, a`) as input and return two Q-values as outputs
      const targetNextStateValues1_1 = agent.critic_target.call(nextStates1, clipActions1).squeeze([1]);
      const targetNextStateValues1_2 = agent.critic_target2.call(nextStates1, clipActions1).squeeze([1]);
      
      // STEP 8: we keep the minimum of the two Q-values. min(Qt1, Qt2)
      const nextTargetStateValue1 = tf.minimum(targetNextStateValues1_1, targetNextStateValues1_2);
    
      // STEP 9: we get the final target of the two Critic models (Qt = r + gamma * min(Qt1, Qt2) * dones, where gamma is the discount factor)
      const targetValues1 = tf.add(rewards1, tf.mul(tf.scalar(agent.gamma), tf.mul(nextTargetStateValue1, dones1)));
      
      const criticValue1 = agent.critic_main.call(states1, actions1).squeeze([1]); 
      
      const criticLoss1 = tf.losses.meanSquaredError(criticValue1, targetValues1);
      
      
      if (utilsAI.doneFlag) {utilsAI.criticLoss1 = criticLoss1.array();}
      return criticLoss1;
    };

    function lossFunction2() { 
      
      const targetActions2 = agent.actor_target.call(nextStatesTensor, {batchSize: agent.batch_size});
      const nextStates2 = tf.clone(nextStatesTensor); 
      const rewards2 = tf.clone(rewardsTensor);
      const dones2 = tf.clone(donesTensor);
      const states2 = tf.clone(statesTensor);
      const actions2 = tf.clone(actionsTensor);

      const ranNormalActions2 = tf.add(targetActions2, tf.clipByValue(tf.randomNormal(targetActions2.shape, 0.0, 0.2), -0.5, 0.5));
      const clipActions2 = tf.mul(tf.scalar(agent.max_action), tf.clipByValue(ranNormalActions2, agent.min_action, agent.max_action));
    
      const targetNextStateValues2_1 = agent.critic_target.call(nextStates2, clipActions2).squeeze([1]);
      const targetNextStateValues2_2 = agent.critic_target2.call(nextStates2, clipActions2).squeeze([1]);
      const nextTargetStateValue2 = tf.minimum(targetNextStateValues2_1, targetNextStateValues2_2);
      const targetValues2 = tf.add(rewards2, tf.mul(tf.scalar(agent.gamma), tf.mul(nextTargetStateValue2, dones2)));
      
    const criticValue2 = agent.critic_main2.call(states2, actions2).squeeze([1])
    const criticLoss2 = tf.losses.meanSquaredError(criticValue2, targetValues2);
    
    if (utilsAI.doneFlag) {utilsAI.criticLoss2 = criticLoss2.array();}
    return criticLoss2
    };

    // STEP 12: we backpropigate the Critic loss and update the parameters of the critic models through optiizers
    const gWeights1 = this.critic_main.model.getWeights(true);
   
    this.critic_main.model.optimizer.minimize(lossFunction1,gWeights1);

    const gWeights2 = this.critic_main2.model.getWeights(true);
  
    this.critic_main2.model.optimizer.minimize(lossFunction2,gWeights2);
    
    this.trainstep += 1;

    // STEP 13: once every two iterations, we update the Actor model by performing gradient ascent on the output of the first critic model
    if (this.trainstep % this.actor_update_steps === 0) {
      
      function lossFunction3() {
        const states3 = tf.clone(statesTensor);
        // gradient ascent is the negative of gradient decent.
        //console.log(`statesTensor: ${statesTensor}, ${statesTensor.shape}`);
        const actorCall = agent.actor_main.call(states3, {batchSize: agent.batch_size});
        //console.log(`actorCall: ${actorCall}, ${actorCall.shape}`);
        const criticCall = agent.critic_main.call(states3, actorCall).neg();
        //console.log(`criticCall: ${criticCall}, ${criticCall.shape}`);
        const actorLoss = tf.mean(criticCall);
       if (utilsAI.doneFlag) {utilsAI.actorLoss = actorLoss.array();}
        //console.log(`Actor Loss: ${actorLoss}`);
        
        return actorLoss;
      }
      
      const gWeights3 = this.actor_main.model.getWeights(true);
      
      this.actor_main.model.optimizer.minimize(lossFunction3,gWeights3);

       this.updateTargets();
    }
    // STEP 14/15... updating weights 
    
  }); // End Tidy
  }
  
  async downloadModels() {
    try {
      async function dlGroup1() {
        await agent.actor_main.model.save('downloads://actor_main-model');
        await agent.actor_target.model.save('downloads://actor_target-model');
        await agent.critic_main.model.save('downloads://critic_main-model');

        setTimeout(dlGroup2, 2000);
      }
      async function dlGroup2() {
        await agent.critic_target.model.save('downloads://critic_target-model');
        await agent.critic_main2.model.save('downloads://critic_main2-model');
        await agent.critic_target2.model.save('downloads://critic_target2-model');
      }
      await Promise.all([dlGroup1()]);
      
      } catch (error) {
        console.error(`Error downloading models: ${error}`);
      }
  }
  async downloadActorModel() {
    try {
    await agent.actor_main.model.save('downloads://actor_main-model');
    
    } catch (error) {
      console.error(`Error downloading actor model: ${error}`);
    }


    
  }
  async downloadMemory() {
    try{
      let memSize = agent.memory.cnt >= agent.memory.maxsize ? agent.memory.maxsize : agent.memory.cnt;
      //const actClone = tf.clone(agent.memory.action_memory);
      const stateClone = [];
      const n_stateClone = [];
      const actClone = [];
      const rewardClone = [];
      const doneClone = [];
      for (let i = 0; i < memSize; i++) {
        stateClone.push([...agent.memory.state_memory[i]]);
        n_stateClone.push([...agent.memory.next_state_memory[i]]);
        actClone.push([...agent.memory.action_memory[i]]);
        rewardClone.push(JSON.parse(JSON.stringify(agent.memory.reward_memory[i])));
        doneClone.push(JSON.parse(JSON.stringify(agent.memory.done_memory[i])));
      }
      
      const memoryData = {
      version: JSON.parse(JSON.stringify(Game.version)), // float
      cnt: JSON.parse(JSON.stringify(agent.memory.cnt)), // integer
      memSize: memSize,
      //maxsize: JSON.parse(JSON.stringify(agent.memory.maxsize)),  // integer
      state_memory: stateClone,  // tensor
      next_state_memory: n_stateClone, // tensor
      action_memory: actClone, // tensor
      reward_memory: rewardClone, // tensor
      done_memory: doneClone // tensor
      }
      //console.log(tensorData);
      // Step 1: Convert the array to a suitable format (e.g., JSON)
      const serializedData = JSON.stringify(memoryData);

      // Step 2: Create a Blob
      const blob = new Blob([serializedData], { type: 'application/json' });

      // Step 3: Create an object URL
      const objectURL = URL.createObjectURL(blob);

      // Step 4: Create an anchor element
      const a = document.createElement('a');
      a.href = objectURL;
      a.download = 'memory_data.json'; // Set the desired filename

      // Step 5: Simulate a click to trigger the download
      a.click();
      // Optionally, revoke the object URL after the download is initiated
      URL.revokeObjectURL(objectURL);
    } catch (error) {
      console.error(`Failed to save Memory: ${error}`);
    }
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
              UI.testAgentButton.disabled = false;
              //UI.testMarathonButton.disabled = false;
            }
            else {throw`actor_main weights file not found. Expecting: actor_main-model.weights.bin`}
          }
        break;
        case "actor_target-model.json": {
          let weights;
          for (let j=0; j < modelFiles.length; j++) {
            if (modelFiles[j].name === "actor_target-model.weights.bin") {
              weights = modelFiles[j];
            }
          } // end j loop
          if (weights) {
            agent.actor_target.model = await tf.loadLayersModel(tf.io.browserFiles(
              [modelFiles[i], weights]));
            agent.actor_target.model.compile({optimizer: agent.a_opt, loss: tf.losses.meanSquaredError});
            console.log(`Loaded actor_target model`);
            const index = agent.loadedFiles.indexOf('actor_target');
            if (index !== -1) {agent.loadedFiles.splice(index, 1);}
            agent.loadedFiles.push(`actor_target`);
          }
          else {throw`actor_target weights file not found. Expecting: actor_target-model.weights.bin`}
        }
        break; 
        case "critic_main-model.json": {
          let weights;
          for (let j=0; j < modelFiles.length; j++) {
            if (modelFiles[j].name === "critic_main-model.weights.bin") {
              weights = modelFiles[j];
            }
          } // end j loop
          if (weights) {
            agent.critic_main.model = await tf.loadLayersModel(tf.io.browserFiles(
              [modelFiles[i], weights]));
            agent.critic_main.model.compile({optimizer: agent.c_opt1, loss: tf.losses.meanSquaredError});
            console.log(`Loaded critic_main model`);
            const index = agent.loadedFiles.indexOf('critic_main');
            if (index !== -1) {agent.loadedFiles.splice(index, 1);}
            agent.loadedFiles.push(`critic_main`);
          }
          else {throw`critic_main weights file not found. Expecting: critic_main-model.weights.bin`}
        }
        break;  
        case "critic_target-model.json": {
          let weights;
          for (let j=0; j < modelFiles.length; j++) {
            if (modelFiles[j].name === "critic_target-model.weights.bin") {
              weights = modelFiles[j];
            }
          } // end j loop
          if (weights) {
            agent.critic_target.model = await tf.loadLayersModel(tf.io.browserFiles(
              [modelFiles[i], weights]));
            agent.critic_target.model.compile({optimizer: agent.c_opt1, loss: tf.losses.meanSquaredError});
            console.log(`Loaded critic_target model`);
            const index = agent.loadedFiles.indexOf('critic_target');
            if (index !== -1) {agent.loadedFiles.splice(index, 1);}
            agent.loadedFiles.push(`critic_target`);
          }
          else {throw`critic_target weights file not found. Expecting: critic_target-model.weights.bin`}
        }
        break;  
        case "critic_main2-model.json": {
          let weights;
          for (let j=0; j < modelFiles.length; j++) {
            if (modelFiles[j].name === "critic_main2-model.weights.bin") {
              weights = modelFiles[j];
            }
          } // end j loop
          if (weights) {
            agent.critic_main2.model = await tf.loadLayersModel(tf.io.browserFiles(
              [modelFiles[i], weights]));
            agent.critic_main2.model.compile({optimizer: agent.c_opt2, loss: tf.losses.meanSquaredError});
            console.log(`Loaded critic_main2 model`);
            const index = agent.loadedFiles.indexOf('critic_main2');
            if (index !== -1) {agent.loadedFiles.splice(index, 1);}
            agent.loadedFiles.push(`critic_main2`);
          }
          else {throw`critic_main2 weights file not found. Expecting: critic_main2-model.weights.bin`}
        }
        break;  
        case "critic_target2-model.json": {
          let weights;
          for (let j=0; j < modelFiles.length; j++) {
            if (modelFiles[j].name === "critic_target2-model.weights.bin") {
              weights = modelFiles[j];
            }
          } // end j loop
          if (weights) { 
            agent.critic_target2.model = await tf.loadLayersModel(tf.io.browserFiles(
              [modelFiles[i], weights]));
            agent.critic_target2.model.compile({optimizer: agent.c_opt2, loss: tf.losses.meanSquaredError});
            console.log(`Loaded critic_target2 model`);
            const index = agent.loadedFiles.indexOf('critic_target2');
            if (index !== -1) {agent.loadedFiles.splice(index, 1);}
            agent.loadedFiles.push(`critic_target2`);
          }
          else {throw`critic_target2 weights file not found. Expecting: critic_target2-model.weights.bin`}
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
  }
  loadMemory(file) { // .squeeze([1])
    
    if (file) {
      if (file.name != "memory_data.json") {
        let ignorName = confirm("The memory name doesn't match the default. Continue anyway?")
          if (!ignorName) {return}
      }
    const reader = new FileReader();
    
    reader.onload = function (e) {
      try {
        const parsedData = JSON.parse(e.target.result);
        if (parsedData.memSize > agent.memory.maxsize) {
          throw `memory RBuffer size not large enough to load this memory. 
          memsize:${parsedData.memSize}, Memory Buffer max: ${agent.memory.maxsize}`;
        }
        if (parsedData.version != Game.version) {
          let ignorVer = confirm("The memory version and program version don't match. Continue anyway?")
          if (!ignorVer) {return}
        }
        
        for (let i = 0; i < parsedData.memSize; i++) {
          const state = parsedData.state_memory[i];
          const next_state = parsedData.next_state_memory[i];
          const action = parsedData.action_memory[i][0]; // [0] cuts off the outer [] because savexp adds it back on
          const reward = parsedData.reward_memory[i];
          let isDone = parsedData.done_memory[i];
          isDone = isDone == 0 ? 1 : 0;   // Must flip back to orignal because savexp flips it

          agent.savexp(state, next_state, action, isDone, reward);
        }
        agent.memory.cnt = parsedData.cnt;
        console.log(`Loaded memory succesfully with count: ${parsedData.cnt}`);
        //console.log(agent.memory);
        UI.memoryLoadedInfo.innerHTML = `Memory Loaded: ${file.name}`;
        UI.memoryLoadedInfo.style="color:rgb(10, 218, 10)";
        UI.trainFromMemoryButton.disabled = false;
        
      } catch (error){
        console.error('Error parsing JSON or storing memory:', error);
      }
    };

    reader.readAsText(file);
  }
  }
  
} // End Agent Class

function envReset() {
  observationSpace.initUpdate();
  const os = observationSpace;
  
  if (UI.randZomCheckbox.checked) {
    let zomLoc = Game.getZombieSpawn(); // gives center
    os.zomCoords[0] = zomLoc[0];
    os.zomCoords[1] = zomLoc[1];

    zom1.x = zomLoc[0] - (zom1.width/2);
    zom1.y = zomLoc[1] - (zom1.height/2);
    
  }
  
  /*
  if (UI.randCivCheckbox.checked || Game.marathon) {
    const ents = Game.entities;
  for (let i = ents.length-1; i >= 0; i--) {
    if (ents[i].type == "player") {continue}
    let civLoc = Game.getCivilianSpawn();
      let x = civLoc[0] - 10;
      let y = civLoc[1] - 10;
      ents[i].x = x;
      ents[i].y = y;

      if (ents[i] === civ1) {
        os.civCoords[0] = civLoc[0];
        os.civCoords[1] = civLoc[1];
      }
  }

   // let civLoc = Game.getCivilianSpawn(); // gives center
    
    //os.defaults[0][os.locNums.zX] = parseFloat((civLoc[0] * os.xyNorm).toFixed(8)); // normalize
    //os.defaults[0][os.locNums.zY] = parseFloat((civLoc[1] * os.xyNorm).toFixed(8));

   // civ1.x = civLoc[0] - (civ1.width/2);
    //civ1.y = civLoc[1] - (civ1.height/2);
    
  }
  */
  if (UI.randAgentCheckbox.checked || Game.marathon) {
    
    let center = [os.zomCoords[0], os.zomCoords[1]];
    let agentLoc = Game.getAgentSpawn(center,"zombie"); // gives center
    os.agentCoords[0] = agentLoc[0];
    os.agentCoords[1] = agentLoc[1];
    
    const baseX = agentLoc[0] * os.xyNorm;
    const baseY = agentLoc[1] * os.xyNorm;
    const floorX = Math.floor(baseX * agent.decScale);
    const floorY = Math.floor(baseY * agent.decScale);

    os.defaults[0][os.locNums.aX] = (floorX / agent.decScale);
    os.defaults[0][os.locNums.aY] = (floorY / agent.decScale);

    player.x = agentLoc[0] - (player.width/2);
    player.y = agentLoc[1] - (player.height/2);
  }

  //const civDrawX = os.civCoords[0] - (civ1.width/2); // civCoords is center. These are used for drawing
  //const civDrawY = os.civCoords[1] - (civ1.height/2);
  //Game.civilianMoves.push([civDrawX, civDrawY]);

  // Reset the state space
  os.stateSpace = JSON.parse(JSON.stringify(os.defaults));
  os.next_stateSpace = JSON.parse(JSON.stringify(os.defaults));
  
  //const state = JSON.parse(JSON.stringify(os.stateSpace));
  
  //return state;
}

function envStep(action, currentStep) {
  const actionClone = JSON.parse(JSON.stringify(action));
  const OS = observationSpace;
  const osNS = observationSpace.next_stateSpace;
  const LN = observationSpace.locNums
  const ents = Game.entities;
  let hitWall = false;
  let isDone = false;
  let reward = 0;
  let penalty = 0;
  
   const x = player.speed * actionClone[0]; 
   const y = player.speed * actionClone[1];

  const movementThreshold = 0; // 0.15 then 0.05 then 0
  // move X
  if ((actionClone[0] < -movementThreshold) || (actionClone[0] > movementThreshold)) {
    if ((OS.agentCoords[0] + x) < (player.width/2)) { // hit left wall
      OS.agentCoords[0] = (player.width/2);  // xyNorm based on mapsize of 500 is 0.002. *2-1 normalizes to (-1,1)
      const base = (player.width / 2) * OS.xyNorm;
      const floorX = Math.floor(base * agent.decScale); // 100000 (100,000)
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

  const zomAction = zom1.chase(); // [x,y]
  OS.zomCoords[0] = zomAction[0];
  OS.zomCoords[1] = zomAction[1];

   // can return the polygon points
  for (let i = ents.length -1; i >= 0; i--) {
    if (!ents[i] || ents[i].type == "player") {continue}
    ents[i].createPolygon();
  }

  //let civDist = utilsAI.distance(OS.agentCoords[0], OS.agentCoords[1], OS.civCoords[0], OS.civCoords[1]);
  //let civAngle = utilsAI.angle([OS.agentCoords[0], OS.agentCoords[1]], [OS.civCoords[0], OS.civCoords[1]]);
  const zomDist = utilsAI.distance(OS.agentCoords[0], OS.agentCoords[1], OS.zomCoords[0], OS.zomCoords[1]);
  //let zomAngle = utilsAI.angle([OS.agentCoords[0], OS.agentCoords[1]], [OS.zomCoords[0], OS.zomCoords[1]]);
  //let agentHeading = utilsAI.angle([OS.stateSpace[0][LN.aX], OS.stateSpace[0][LN.aY]], [OS.agentCoords[0], OS.agentCoords[1]]); 

  //let detectedEnts = []; // for multiple or marathon

  player.sensor.update(Game.mapBorders,Game.entities); // hardcoded zombie
    const offsetsXY = player.sensor.readings.map((reading) => {
      if (reading == null) {
          return null;  // Default value for no detection
      } else { 
          // Adjust the offset based on the type of the detected object
          if (reading.ent === "zombie") {
              
              return {xy:[reading.x,reading.y],type:"zombie"};

          } else if (reading.ent === "wall") { 

              return {xy:[reading.x,reading.y],type:"wall"};
              //return [reading.x,reading.y];
          } else {
              //return [reading.x,reading.y]; // Default value for unknown types
              return null;
          }
      }
  });
 
  for (let i = 0; i < offsetsXY.length; i++){
    let distance;
    
    if (offsetsXY[i] == null) {distance = 0}
    else {
      distance = utilsAI.distance(OS.agentCoords[0], OS.agentCoords[1], offsetsXY[i].xy[0], offsetsXY[i].xy[1]);
      distance *= OS.xyNorm; //scaled (0,1) descending
      if (offsetsXY[i].type == "zombie") {distance = -(1 - distance);} // (-1,0) descending for zombie
      else {distance = 1 - distance} //scaled (0,1) ascending for walls
      
      distance = Math.floor(distance * agent.decScale);
      distance /=  agent.decScale;
      
      
    }
    
    osNS[0][i+2] = distance;
  }
 
  //offsetTotal /= offsets.length; // 45 rays
  //offsetTotal /= 2;
  //plusTotal /= 22;
  //minusTotal /= 22; // picked based on testing.
  
  //reward += plusTotal;
 //penalty += minusTotal;
 
  /*
  if (Game.marathon && closestEnt) {
    OS.civCoords[0] = closestEnt.x + 10;
    OS.civCoords[1] = closestEnt.y + 10;
  }
  */

 
const cornerSize = 100;

  if (OS.agentCoords[0] < cornerSize && OS.agentCoords[1] < cornerSize) { // Making the corners negative zones to prevent getting stuck in corners
    hitWall = true;
  }
  else if (OS.agentCoords[0] > (Game.width - cornerSize) && OS.agentCoords[1] < cornerSize) {
    hitWall = true;
  }
  else if (OS.agentCoords[0] > (Game.width - cornerSize) && OS.agentCoords[1] > (Game.height - cornerSize)) {
    hitWall = true;
  }
  else if (OS.agentCoords[0] < cornerSize && OS.agentCoords[1] > (Game.height - cornerSize)) {
    hitWall = true;
  }

const safeDistance = 100; // 500 is ray length. zombie max spawn: map.width / player.speed (100)
const survivalBonus = 1;
//const distPenalty = 0.5;
 reward += survivalBonus; // Every step gets a bonus for survival

  let distanceBase = zomDist;
  let distanceScaling = observationSpace.xyNorm; // (0, 1) xyNorm is hardcoded 0.002 based on map size
  distanceBase *= distanceScaling;
  let distancePenalty = 1 - distanceBase; // invert so it's higher when closer
  distancePenalty /= 2; // (0, 0.5) // share space
  distancePenalty = Math.min(distancePenalty, 0.5); // safeguard
  

  //if (distanceBase > 0.3 && zomDist >= OS.rawDist) {reward += 1}


if (zomDist > safeDistance){ // Reward for simply staying away from zombie.
  //if (!hitWall) {reward += survivalBonus}
  //else {reward += (survivalBonus / 2); }
  
}
else {
  penalty += distancePenalty;
}
//else if (zomDist < (safeDistance/2)){penalty += 0.5} // penalty for being to close to zombie

/*
else {
  const agentSpeed = Math.abs(x) + Math.abs(y);
  if ( !(OS.rawDist <= zomDist && agentSpeed >= zom1.speed)){ penalty += distPenalty;} // only apply distance penalty if agent isn't moving away from zombie
  
}
*/
/*
   //1rad × 180/π = 57.296°
  //1° × π/180 = 0.01745rad
function calculateTimePenalty(step, maxSteps) {
  const maxPenalty = -0.2; // Adjust as needed
  const timeRatio = step / maxSteps;
  const timePenalty = Math.max(maxPenalty, maxPenalty * timeRatio);
  return timePenalty;
}
*/
  
  //else if (civDist < OS.rawDist) {reward += 0.1}
  //if (civDist < player.sensor.rayLength) {
    //penalty += anglePenalty + distancePenalty;

    //if (civDist < OS.rawDist) { 
     // reward += 1;
    //}
    //else {reward += 1}
//}

  //OS.rawDist = civDist;
  //OS.rawAngle = civAngle;
  //OS.rawAngleDiff = anglePenalty;
  OS.rawDist = zomDist;
  //osNS[0][LN.dist] = parseFloat((zomDist * OS.xyNorm).toFixed(3));
  //osNS[0][LN.angle] = parseFloat((zomAngle / Math.PI).toFixed(8));

  //const timePenalty = calculateTimePenalty(currentStep, agent.maxStepCount);
  //const timeBonus = calculateTimeBonus(currentStep, agent.maxStepCount);
  //reward += timeBonus;
  /*
  if (detect) { // only give rewards and penalties if ray detect something.
    reward = reward - penalty;
  }
  else {
    reward = 0;
    penalty = 0;
  }
  */

  if (hitWall) { // penalty for walking into walls
    //penalty += 0.5
    penalty += 1;
    //if (!Game.testing){isDone = true;}
  } 
  //reward -= Math.abs(timePenalty);
  if (penalty > 2) {penalty = 2} // failsafes (-1,1)
  if (reward > 2) {reward = 2}
  reward = reward - penalty;
  reward = Math.floor(reward * agent.decScale);
  reward /=  agent.decScale;

  
  if (zomDist <= player.width) { // finding target overules everything.
      reward = -1; 
      isDone = true; // Original paper says to only use this if terminal state is not horizon (max steps)
  }

  else if ( currentStep >= agent.maxStepCount) {
  
    //reward += 1; // might cause location sticking I think
    console.warn("AGENT SURVIVED!");
    Game.agentWins++;
    UI.agentWins.innerHTML = `Times Won: ${Game.agentWins}`;
    Game.winsThisRun++;
    UI.winsThisRun.innerHTML = `Wins This Run: ${Game.winsThisRun}`;
    //isDone = true;
  }
    
  //console.log(`Step: ${currentStep}, reward: ${reward}`);
  
  const next_State = JSON.parse(JSON.stringify(observationSpace.next_stateSpace));
  //console.log(next_State);
  let terminalFlag = false;
  if (isDone || (currentStep >= agent.maxStepCount)) {terminalFlag = true;}
  const aX = OS.agentCoords[0];
  const aY = OS.agentCoords[1];
  const zX = OS.zomCoords[0];
  const zY = OS.zomCoords[1];
  Game.agentMoves.push([aX, aY, terminalFlag]); // For drawing agent paths
  Game.zombieMoves.push([zX, zY]); // For drawing zombie paths
  return { next_state: next_State, reward: reward, isDone: isDone };
} // end envStep

const agent = new Agent(actionSpace.numberActions, observationSpace.defaults[0].length, observationSpace.defaults[0].length + actionSpace.numberActions); 
let episodes = 5;
const epReward = [];
const totalAvgReward = [];

function main(epNum,stepSize,batchSize,warmupSteps,gamma) {
  console.log("TD3 Started");
  let target = false;
  if (epNum && !isNaN(epNum) && epNum > 0) {episodes = epNum}
  if (stepSize && !isNaN(stepSize) && stepSize > 0) {agent.maxStepCount = stepSize}
  if (batchSize && !isNaN(batchSize) && batchSize > 0) {agent.batch_size = batchSize}
  if (warmupSteps && !isNaN(warmupSteps) && warmupSteps > 0) {agent.warmup = warmupSteps}
  if (gamma && !isNaN(gamma) && gamma >= 0) {agent.gamma = gamma}
  utilsAI.doneFlag = false;

  for (let s = 0; s < episodes; s++) { 
    if (!Game.running) {break}
    if (target) {
      break;
    }
    if (s >= episodes-1) {utilsAI.doneFlag = true;}
    
    // STEP 0: reset everything
    let totalReward = 0;
    envReset(); 
    let done = false;
    let currentStep = 1;
    let stepDone = false;
    
    while (!done) {
      if (!Game.running) {break} 
      
      // STEP 1a: get an action based on the current state 
      let state = JSON.parse(JSON.stringify(observationSpace.stateSpace));
      const action = agent.act(state);
      
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
      
      // STEP 2a: step the environment with the action, returning the new state, rewards, and if done
      const { next_state, reward, isDone } = envStep(clipAction, currentStep);
      
      // STEP 3: save the new state to the memory buffer
      
      if (!Game.testing){ 
        
        agent.savexp(state, next_state, clipAction, isDone, reward);
        // Step 4-15..: train the system
        agent.train(); // only gets called if memory.cnt >= agent.batch size
      }
      
      // STEP 16: make the current state the new state
      observationSpace.stateSpace = JSON.parse(JSON.stringify(observationSpace.next_stateSpace)); // DEEP COPY
      
      totalReward += reward;
      
      if (currentStep >= agent.maxStepCount) {stepDone = true}
      currentStep++

      if (isDone || stepDone) {
        epReward.push(JSON.parse(JSON.stringify(totalReward)));
        const avgReward = epReward.slice(-100).reduce((a, b) => a + b, 0) / Math.min(epReward.length, 100);
        totalAvgReward.push(avgReward);

        console.log(`Total reward at Episode ${s} is ${totalReward}. avg reward: ${avgReward}`);

        //if (Math.floor(avgReward) === 100) {
        //  target = true;
        //}
        done = true;
        if (!Game.testing){
          Game.episodesRan++;
          UI.episodesRan.innerHTML = `Episodes Ran: ${Game.episodesRan}`;
        }
        
      }
    }
  }
  animateAgent();
}
