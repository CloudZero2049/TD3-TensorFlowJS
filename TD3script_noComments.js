
const utilsAI = {
  distance: function(x1,y1,x2,y2) {
    return Math.floor(Math.hypot(x2 - x1, y2 - y1));
  },
  angle: function(from,to) { 
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

  initUpdate: function() {
    observationSpace.defaults[0][0] = player.x + (player.width/2);
    observationSpace.defaults[0][1] = player.y + (player.height/2);
    observationSpace.civLoc[0] = civ1.x + (civ1.width/2);
    observationSpace.civLoc[1] = civ1.y + (civ1.height/2);
  },
  civLoc:[civ1.x + (civ1.width/2), civ1.y + (civ1.height/2)],
  defaults: [[player.x + (player.width/2), player.y + (player.height/2), 28, 1.56]], 
  stateSpace: [[player.x + (player.width/2), player.y + (player.height/2), 28, 1.56]],
  next_stateSpace: [[player.x + (player.width/2), player.y + (player.height/2), 28, 1.56]],
}

const actionSpace = {
  numberActions: 2,
  shape: [2], // not used
  actions: [0,0], // not used
  low: [-1,-1],
  high: [1,1]
};

class RBuffer {
  constructor(maxsize, statedim, n_actions) { 
    this.cnt = 0;
    this.maxsize = maxsize; 
    this.state_memory = tf.zeros([maxsize, ...statedim]); 
    this.next_state_memory = tf.zeros([maxsize, ...statedim]);
    this.action_memory = tf.zeros([maxsize, n_actions]);
    this.reward_memory = tf.zeros([maxsize]);
    this.done_memory = tf.zeros([maxsize]); 
  }
  
  storexp(state, next_state, action, done, reward) { 
    const index = this.cnt % this.maxsize;
    done = done ? 1 : 0; 

    this.state_memory[index] = JSON.parse(JSON.stringify(state));
    this.next_state_memory[index] = JSON.parse(JSON.stringify(next_state));
    this.action_memory[index] = JSON.parse(JSON.stringify([action])); 
    this.reward_memory[index] = JSON.parse(JSON.stringify(reward));
    this.done_memory[index] = JSON.parse(JSON.stringify(1-done)); 
    this.cnt += 1;
  }

  sample(batch_size) {
    const max_mem = Math.min(this.cnt, this.maxsize);
   
    let batch = [];
    for (let i = 0; i < batch_size; i++) {
        let index;
        do {
            index = Math.floor(Math.random() * max_mem);
        } while (batch.includes(index));
        batch.push(index);
    }

    const states = batch.map(index => this.state_memory[index]);
    const next_states = batch.map(index => this.next_state_memory[index]);
    const rewards = batch.map(index => this.reward_memory[index]);
    const actions = batch.map(index => this.action_memory[index]);
    const dones = batch.map(index => this.done_memory[index]);
    
    return [states, next_states, rewards, actions, dones];
  }
}

class Critic {  
  constructor(inputShape) { 
    this.model = tf.sequential();
    this.model.add(tf.layers.dense({ inputShape: [inputShape], units: 400, activation: 'relu', dtype: 'float32', kernelInitializer: 'randomNormal'})); 
    this.model.add(tf.layers.dense({ units: 300, activation: 'relu', dtype: 'float32', kernelInitializer: 'randomNormal' }));
    this.model.add(tf.layers.dense({ units: 1, activation: null, dtype: 'float32', kernelInitializer: 'randomNormal' })); 
  }

  call(inputstate, action) { 
  
    const catTensor = tf.concat([inputstate, action], 1);

    const x = this.model.predict(catTensor);
    
    return x;
  
  }
}

class Actor {
  constructor(n_actions,inputShape) { 
    this.model = tf.sequential();
    this.model.add(tf.layers.dense({ inputShape: inputShape, units: 400, activation: 'relu', dtype: 'float32', kernelInitializer: 'randomNormal'  })); 
    this.model.add(tf.layers.dense({ units: 300, activation: 'relu', dtype: 'float32', kernelInitializer: 'randomNormal'   }));
    this.model.add(tf.layers.dense({ units: n_actions, activation: 'tanh', dtype: 'float32', kernelInitializer: 'randomNormal'  })); 

  }
  
  call(stateTensor) {
   
    const pred = this.model.predict(stateTensor);
   
    return pred;
   
  }
}

class Agent {
  constructor(n_actions = 2, inputShapeA = 4, inputShapeC = 6, alpha = 0.001, beta = 0.002, gamma = 0.99, tau = 0.005, warmup = 64, RBufferSize = 100000) {                           
    this.actor_main = new Actor(n_actions,inputShapeA);
    this.actor_target = new Actor(n_actions,inputShapeA);
    this.critic_main = new Critic(inputShapeC);
    this.critic_main2 = new Critic(inputShapeC);
    this.critic_target = new Critic(inputShapeC);
    this.critic_target2 = new Critic(inputShapeC);
    this.batch_size = 32;
    this.n_actions = n_actions;
    this.a_opt = tf.train.adam(alpha);
    this.c_opt1 = tf.train.adam(beta);
    this.c_opt2 = tf.train.adam(beta);
    this.memory = new RBuffer(RBufferSize, [observationSpace.stateSpace[0].length], this.n_actions); 
    this.gamma = gamma;
    this.tau = tau;
    this.actor_update_steps = 2; 
    this.warmup = warmup; 
    this.trainstep = 0;
    this.maxStepCount = 32; // not tied to trainstep
    this.min_action = actionSpace.low[0];  
    this.max_action = actionSpace.high[0]; 

    //video sets noise here. and update_network_parameters(tau=1)
    this.critic_main.model.compile({optimizer: this.c_opt1, loss: tf.losses.meanSquaredError}); 
    this.critic_main2.model.compile({optimizer: this.c_opt2, loss: tf.losses.meanSquaredError}); 
    this.actor_main.model.compile({optimizer: this.a_opt, loss: tf.losses.meanSquaredError}); 
    this.critic_target.model.compile({optimizer: this.c_opt1, loss: tf.losses.meanSquaredError});
    this.critic_target2.model.compile({optimizer: this.c_opt2, loss: tf.losses.meanSquaredError}); 
    this.actor_target.model.compile({optimizer: this.a_opt, loss: tf.losses.meanSquaredError}); 
    
    this.updateTarget(1); 
  }

  act(state, evaluate = false) { 
  if (this.trainstep > this.warmup) {
    evaluate = true;
  }
  let returnValue = tf.tidy(() => {     
  
  const stateTensor = tf.tensor(state); 
  let actions = this.actor_main.call(stateTensor);

  if (!evaluate) { 
    const noise = tf.randomNormal(actions.shape, 0.0, 0.1); 
  
    actions = tf.add(actions, noise);
    
  }
  
  const clippedActions = tf.clipByValue(actions, this.min_action, this.max_action); 
 
  const scaledActions = tf.mul(clippedActions, tf.scalar(this.max_action));
 
  return scaledActions.arraySync()[0]; 
}); 
return returnValue;
}
 
  savexp(state, next_state, action, done, reward) { 
    this.memory.storexp(state, next_state, action, done, reward);
  }

  updateTarget(tau = null) { 
    if (tau == null) {
      tau = this.tau;
    }
    
    // STEP 14: we update the weights of the Actor target by polyak averaging
    const weights1 = [];
  
    const targets1 = this.actor_target.model.getWeights().map(tf.clone);
    
    const mainWeights1 = this.actor_main.model.getWeights().map(tf.clone);
 
    for (let i = 0; i < mainWeights1.length; i++) {
      const updatedWeight1 = tf.tidy(() => {
        const weightedMain1 = tf.mul(mainWeights1[i], tf.scalar(tau));
       
        const weightedTarget1 = tf.mul(targets1[i], tf.scalar(1 - tau));
       
        const combinedWeight1 = tf.add(weightedMain1, weightedTarget1);
      
        return combinedWeight1;
      }); 
      
      weights1.push(updatedWeight1);
      
    };
    this.actor_target.model.setWeights(weights1);
   
    const weights2 = [];
    const targets2 = this.critic_target.model.getWeights().map(tf.clone);
    const mainWeights2 = this.critic_main.model.getWeights().map(tf.clone);
    for (let i = 0; i < mainWeights2.length; i++) {
      const updatedWeight2 = tf.tidy(() => {
        const weightedMain2 = tf.mul(mainWeights2[i], tf.scalar(tau));
        const weightedTarget2 = tf.mul(targets2[i], tf.scalar(1 - tau));
        const combinedWeight2 = tf.add(weightedMain2, weightedTarget2);
        return combinedWeight2;
      }); 
      weights2.push(updatedWeight2);
    } 

    this.critic_target.model.setWeights(weights2);
   
    const weights3 = [];
    const targets3 = this.critic_target2.model.getWeights().map(tf.clone);
    const mainWeights3 = this.critic_main2.model.getWeights().map(tf.clone);
    for (let i = 0; i < mainWeights3.length; i++) {
      const updatedWeight3 = tf.tidy(() => {
        const weightedMain3 = tf.mul(mainWeights3[i], tf.scalar(tau));
        const weightedTarget3 = tf.mul(targets3[i], tf.scalar(1 - tau));
        const combinedWeight3 = tf.add(weightedMain3, weightedTarget3);
        return combinedWeight3;
      }); 
      weights3.push(updatedWeight3);
    } 

    this.critic_target2.model.setWeights(weights3);
   
  }
  // STEPS 4-15...
  train() {
    
    if (this.memory.cnt < this.batch_size) { 
        return; 
    }
    tf.tidy(() => {
    // STEP 4: we sample a batch of transitions (s, s`, a, r) from memory
    const [states, nextStates, rewards, actions, dones] = this.memory.sample(this.batch_size);
      
    const statesTensor = tf.tensor(states).squeeze();
    const nextStatesTensor = tf.tensor(nextStates).squeeze();
    const rewardsTensor = tf.tensor(rewards); //[]
    const actionsTensor = tf.tensor(actions).squeeze(); 
    const donesTensor = tf.tensor(dones) // []

      // Video 2 adds the loss functions together
    function lossFunction1() { 
      const targetActions1 = agent.actor_target.call(nextStatesTensor); 
      const rewards1 = tf.clone(rewardsTensor);
  
      // STEP 6: We add Gaussian noise to the next action a` and we clamp it in a range of values supported by environment
      const ranNormalActions1 = tf.clone(tf.add(targetActions1, tf.clipByValue(tf.randomNormal(targetActions1.shape, 0.0, 0.2), -0.5, 0.5)));//.map(tf.clone); // video 2 uses actionsTensor?
    
      const clipActions1 = tf.clone(tf.mul(tf.scalar(agent.max_action), tf.clipByValue(ranNormalActions1, agent.min_action, agent.max_action)));//.map(tf.clone); 
      
      // STEP 7: The two Critic targets take each the couple (s`, a`) as input and return two Q-values as outputs
      const targetNextStateValues1_1 = agent.critic_target.call(nextStatesTensor, clipActions1).squeeze([1]);
      const targetNextStateValues1_2 = agent.critic_target2.call(nextStatesTensor, clipActions1).squeeze([1]);
     
      // STEP 8: we keep the minimum of the two Q-values. min(Qt1, Qt2)
      const nextTargetStateValue1 = tf.minimum(targetNextStateValues1_1, targetNextStateValues1_2);
     
      // STEP 9: we get the final target of the two Critic models (Qt = r + gamma * min(Qt1, Qt2) * dones, where gamma is the discount factor)
     
      const targetValues1 = tf.add(rewards1, tf.mul(tf.scalar(agent.gamma), tf.mul(nextTargetStateValue1, donesTensor)));
    
      const criticValue1 = agent.critic_main.call(statesTensor, actionsTensor).squeeze([1]); // squeese? says wont learn otherwise
     
      const criticLoss1 = tf.losses.meanSquaredError(criticValue1, targetValues1);
     
      return criticLoss1;
    };

    function lossFunction2() { 
      const targetActions2 = agent.actor_target.call(nextStatesTensor); 
      const rewards2 = tf.clone(rewardsTensor);

      const ranNormalActions2 = tf.clone(tf.add(targetActions2, tf.clipByValue(tf.randomNormal(targetActions2.shape, 0.0, 0.2), -0.5, 0.5)));
      const clipActions2 = tf.clone(tf.mul(tf.scalar(agent.max_action), tf.clipByValue(ranNormalActions2, agent.min_action, agent.max_action)));
    
      const targetNextStateValues2_1 = agent.critic_target.call(nextStatesTensor, clipActions2).squeeze([1]);
      const targetNextStateValues2_2 = agent.critic_target2.call(nextStatesTensor, clipActions2).squeeze([1]);
      const nextTargetStateValue2 = tf.minimum(targetNextStateValues2_1, targetNextStateValues2_2);
      const targetValues2 = tf.add(rewards2, tf.mul(tf.scalar(agent.gamma), tf.mul(nextTargetStateValue2, donesTensor)));

    const criticValue2 = agent.critic_main2.call(statesTensor, actionsTensor).squeeze([1]) 
    const criticLoss2 = tf.losses.meanSquaredError(criticValue2, targetValues2);
  
    return criticLoss2
  
    };

    
    // STEP 12: we backpropigate the Critic loss and update the parameters of the critic models through optiizers
    const gWeights1 = this.critic_main.model.getWeights(true)//.map(tf.clone); // this does give all the weights right?

    this.critic_main.model.optimizer.minimize(lossFunction1,gWeights1);

    const gWeights2 = this.critic_main2.model.getWeights(true)
    this.critic_main2.model.optimizer.minimize(lossFunction2,gWeights2);
    
    this.trainstep += 1;

    // STEP 13: once every two iterations, we update the Actor model by performing gradient ascent on the output of the first critic model
    if (this.trainstep % this.actor_update_steps === 0) {
      
      function lossFunction3() {
  
        const actorCall = agent.actor_main.call(statesTensor);
   
        const criticCall = agent.critic_main.call(statesTensor, actorCall).neg();
  
        const actorLoss = tf.mean(criticCall);
    
        return actorLoss;

      }
      
      const gWeights3 = this.actor_main.model.getWeights(true)
      this.actor_main.model.optimizer.minimize(lossFunction3,gWeights3);

       // STEP 14/15... updating weights every two iterations. moved here because the original paper says to.

      
    }
    this.updateTarget();
  }); 

  
  }
  // Method for saving the model
  saveModel() {

  }
  // Method for loading a model
  loadModel() {

  }
} // End Agent Class


function normalizeData(data) {
  const inputMax = data.map(feature => Math.max(...feature));
  const inputMin = data.map(feature => Math.min(...feature));

  const normData = data.map((feature, index) =>
    feature.map(value => (value - inputMin[index]) / (inputMax[index] - inputMin[index]))
  );
  return normData
}

function normalizeReward(reward, minRange = -1, maxRange = 1) {
  
  const scaledValue = (x - minRange) / (maxRange - minRange) * 2 - 1;
  return Math.max(-1, Math.min(1, scaledValue));

}

function envReset() {

  observationSpace.stateSpace = JSON.parse(JSON.stringify(observationSpace.defaults));
  observationSpace.next_stateSpace = JSON.parse(JSON.stringify(observationSpace.defaults));
  
  const state = observationSpace.stateSpace;
  
  return state;
}


function envStep(action, state, n_steps) {
  const actionClone = JSON.parse(JSON.stringify(action));
  const playerSpeed = 5;  // temp hardcode 
  const os = observationSpace.next_stateSpace;
  let hitWall = false;

   const x = (playerSpeed * actionClone[0]); 
   const y = (playerSpeed * actionClone[1]); 
 
  // move X
  if (actionClone[0] < -0.001) {
    if ((os[0][0] - Math.abs(actionClone[0])) < 0.001) {os[0][0] = 0.001; hitWall = true;}
    else {os[0][0] += x;}
  }
  else if (actionClone[0] > 0.001) {
    if ((os[0][0] + Math.abs(actionClone[0])) > (Game.width - 20)) {os[0][0] = (Game.width - 20); hitWall = true;}
    else {os[0][0] += x;}
    
  }
  // move Y
  if (actionClone[1] < -0.001) {
    if ((os[0][1] - Math.abs(actionClone[1])) < 0.001) {os[0][1] = 0.001;  hitWall = true;}
    else {os[0][1] += y;}
  }
  else if (actionClone[1] > 0.001) {
    if ((os[0][1] + Math.abs(actionClone[1])) > (Game.height - 20)) {os[0][1] = (Game.height - 20); hitWall = true;}
    else {os[0][1] += y;}
    
  }
  const mx = os[0][0];
  const my = os[0][1];

  Game.agentMoves.push([mx, my])

  let isDone = false;
  let reward = 0;
  
  let civDist = utilsAI.distance(os[0][0],os[0][1],observationSpace.civLoc[0],observationSpace.civLoc[1]);
  let civAngle = utilsAI.angle([os[0][0],os[0][1]],[observationSpace.civLoc[0],observationSpace.civLoc[1]]);
  let angleDeg1 = civAngle * (180 / Math.PI);
  let angleDeg2 = state[0][3] * (180 / Math.PI);
  let angleDegDif = Math.abs(angleDeg1 - angleDeg2);

let angleReward = 180 - angleDegDif; 
let angleRewardScaling = 0.005; // Adjust as needed
angleReward *= angleRewardScaling;

let distanceReward = 1 / civDist; 
let distanceRewardScaling = 0.1; // Adjust as needed
distanceReward *= distanceRewardScaling;
if (civDist > os[0][2]) {distanceReward -= 0.5}
else if (civDist < os[0][2]) {distanceReward += 0.5;}


let totalReward = distanceReward + angleReward;

reward += totalReward; 

 
  console.log(`step: ${n_steps}, civ Dist: ${civDist}`);
  console.log(`distanceReward: ${distanceReward}`);
  console.log(`civAngle: ${civAngle}`);
  console.log(`angleDegDif: ${angleDegDif}`);
  console.log(`angleReward: ${angleReward}`);
  console.log(`total Reward: ${totalReward}`);
 
  os[0][2] = JSON.parse(JSON.stringify(civDist));
  os[0][3] = civAngle;
  if (civDist <= player.width) {reward += 5; console.warn("AGENT FOUND CIVILIAN!");isDone = true;}
  if (hitWall) {reward -= 1}
 
  console.log(`reward: ${reward}`);

  const base_Next_State = JSON.parse(JSON.stringify(observationSpace.next_stateSpace));
  const next_State = normalizeData(base_Next_State);

  if (n_steps >= agent.maxStepCount) {isDone = true}
  return { next_state: next_State, reward: reward, isDone: isDone };
}



const agent = new Agent(actionSpace.numberActions, observationSpace.stateSpace[0].length, observationSpace.stateSpace[0].length + actionSpace.numberActions); 
const episodes = 5; 
const epReward = [];
const totalAvgReward = [];
let target = false;

function main() {  
  observationSpace.initUpdate();
  for (let s = 0; s < episodes; s++) {
    if (!Game.running) {break}
    if (target) {
      break;
    }
    // STEP 0: reset everything
    let totalReward = 0;
    let state = envReset();
    let done = false;
    let n_steps = 0;
    let stepDone = false;
   
    while (!done) {
      if (!Game.running) {break}
  
      const normalState = normalizeData(state);
  
      // STEP 1a: get an action based on the current state 
      const action = agent.act(normalState); // choose_action(observation), is envReset(). // video 2 adds more noise to the action
    
      // STEP 2a: step the environment with the action, returning the new state, rewards, and if done
      const { next_state, reward, isDone } = envStep(action, state, n_steps);
   
      // STEP 3: save the new state to the memory buffer
  
      const normal_Next_State = normalizeData(next_state);
 

      agent.savexp(normalState, normal_Next_State, action, isDone, reward); 
      
      // Step 4-15..: train the system
      agent.train(); 

      // STEP 16: make the current state the new state
      state = JSON.parse(JSON.stringify(observationSpace.next_stateSpace)); // DEEP COPY
      
      totalReward += reward;
    
      
      if (n_steps >= agent.maxStepCount) {stepDone = true} 
      n_steps++
      

      if (isDone || stepDone) {
        epReward.push(JSON.parse(JSON.stringify(totalReward)));
        const avgReward = epReward.slice(-100).reduce((a, b) => a + b, 0) / Math.min(epReward.length, 100);
        totalAvgReward.push(avgReward);

        console.log(`Total reward at step ${s} is ${totalReward} and avg reward is ${avgReward}`);

        if (Math.floor(avgReward) === 100) {
          target = true;
        }
        done = true;
      }
    }
  }
  animateAgent();
}
