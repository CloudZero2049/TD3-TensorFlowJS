
const utilsAI = {
  criticLoss1: 0,
  criticLoss2: 0,
  actorLoss: 0,
  doneFlag: false,
  logLosses: function() { // logs the most recent loss values at end of all episodes
    console.table({critic1:utilsAI.criticLoss1,critic2:utilsAI.criticLoss2,actor:utilsAI.actorLoss});
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
  //[agentX,agentY,civilianX,civilianY,dist to civ, angle to civ]
  initUpdate: function() { // [0,0],[0,0]
    let defs = this.defaults;
  
    this.agentCoords[0] = parseInt(UI.aSliderX.value) + (player.width/2);
    this.agentCoords[1] = parseInt(UI.aSliderY.value) + (player.height/2);
    this.civCoords[0] = parseInt(UI.cSliderX.value) + (civ1.width/2);
    this.civCoords[1] = parseInt(UI.cSliderY.value) + (civ1.height/2);
    //this.zomCoords[0] = parseInt(UI.zSliderX.value) + (zom1.width/2);
    //this.zomCoords[1] = parseInt(UI.zSliderY.value) + (zom1.height/2);
    
    player.x = parseInt(UI.aSliderX.value);
    player.y = parseInt(UI.aSliderY.value);
    //zom1.x = parseInt(UI.zSliderX.value);
    //zom1.y = parseInt(UI.zSliderY.value);
    civ1.x = parseInt(UI.cSliderX.value);
    civ1.y = parseInt(UI.cSliderY.value);

    //this.rawDist = utilsAI.distance(this.agentCoords[0], this.agentCoords[1], this.zomCoords[0], this.zomCoords[1]);
    //this.rawAngle = utilsAI.angle([this.agentCoords[0],this.agentCoords[1]],[this.zomCoords[0], this.zomCoords[1]]);
    const os = this;
    //defs[0][0] = parseFloat((this.agentCoords[0] * this.xyNorm).toFixed(4)); // normalizing values
    //defs[0][1] = parseFloat((this.agentCoords[1] * this.xyNorm).toFixed(4));
    defs[0][0] = (Math.floor((this.agentCoords[0] * os.xyNorm)*100000) / 100000);
    defs[0][1] = (Math.floor((this.agentCoords[1] * os.xyNorm)*100000) / 100000);
    //defs[0][2] = parseFloat((this.zomCoords[0] * this.xyNorm).toFixed(8));
    //defs[0][3] = parseFloat((this.zomCoords[1] * this.xyNorm).toFixed(8));
    //defs[0][27] = parseFloat(((utilsAI.distance(os.agentCoords[0], os.agentCoords[1], os.zomCoords[0], os.zomCoords[1]) * os.xyNorm)).toFixed(3));
    //defs[0][5] = parseFloat(((utilsAI.angle([os.agentCoords[0],os.agentCoords[1]],[os.zomCoords[0], os.zomCoords[1]])+ Math.PI) / (2 * Math.PI)).toFixed(8));
    for (let i = 2; i < defs[0].length; i++) {
      defs[0][i] = 1;
    }
  },
  locNums: {
    aX: 0,
    aY: 1,
    s1: 2,
    s35:36,
    //dist: 27,
    
  },
  agentCoords: [player.x + (player.width/2), player.y + (player.height/2)],
  civCoords: [civ1.x + (civ1.width/2), civ1.y + (civ1.height/2)],
  //zomCoords: [zom1.x + (zom1.width/2), zom1.y + (zom1.height/2)],
  rawDist: utilsAI.distance(player.x + (player.width/2), player.y + (player.height/2), civ1.x + (civ1.width/2), civ1.y + (civ1.height/2)),
  rawAngle: utilsAI.angle([player.x + (player.width/2), player.y + (player.height/2)], [civ1.x + (civ1.width/2), civ1.y + (civ1.height/2)]),
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
  high: [1,1]
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

    const states = batch.map(index => [...this.state_memory[index]]);
    const next_states = batch.map(index => [...this.next_state_memory[index]]);
    //const rewards = batch.map(index => this.reward_memory[index]);
    const rewards = batch.map(index => JSON.parse(JSON.stringify(this.reward_memory[index])));
    const actions = batch.map(index => [...this.action_memory[index]]);
    const dones = batch.map(index => JSON.parse(JSON.stringify(this.done_memory[index])));
    
    return [states, next_states, rewards, actions, dones];
  }
}

class Critic { 
  constructor(inputShape) {
    this.model = tf.sequential(); // {l1:0.01 ,l2:0.01}
    this.model.add(tf.layers.dense({ inputShape: [inputShape], units: 128, activation: 'relu', dtype: 'float32', kernelInitializer: 'heNormal',kernelRegularizer:tf.regularizers.l2({l2:0.1})}));
    this.model.add(tf.layers.dense({ units: 128, activation: 'relu', dtype: 'float32', kernelInitializer: 'heNormal',kernelRegularizer:tf.regularizers.l2({l2:0.1}) }));
    this.model.add(tf.layers.dense({ units: 1, activation: null, dtype: 'float32',kernelRegularizer:tf.regularizers.l2({l2:0.1}) })); // Output to 1 Q value
    
  }

  call(inputstate, actions) { // feed forward
    //console.log(`inputstate: ${inputstate}`);
    //console.log(`action: ${action}`);
   
   // STEP 10b: we concatinate the input state with actio
  // Concatenate the two tensors along the last axis (axis 1)
    const catTensor = tf.concat([inputstate, actions], 1);
   //console.log(`catTensor: ${catTensor.shape}`);
    const pred = this.model.predict(catTensor, {batchSize: agent.batch_size});
    
    return pred;
  }
}
//kernelInitializer: tf.heNormal(shape)
class Actor {
  constructor(n_actions,inputShape) { // has alpha in constructor (video)
    this.model = tf.sequential();
    this.model.add(tf.layers.dense({ inputShape: inputShape, units: 128, activation: 'relu', dtype: 'float32', kernelInitializer: 'heNormal',kernelRegularizer:tf.regularizers.l2({l2:0.1})  }));
    this.model.add(tf.layers.dense({ units: 128, activation: 'relu', dtype: 'float32', kernelInitializer: 'heNormal',kernelRegularizer:tf.regularizers.l2({l2:0.1})   }));
    this.model.add(tf.layers.dense({ units: n_actions, activation: 'tanh', dtype: 'float32', kernelInitializer: 'glorotUniform',kernelRegularizer:tf.regularizers.l2({l2:0.1})  })); // output units: is number of actions/action space
    // from DDPG paper. pi - tangent hyperbolic, +-1
    // if action bounds are say +-2, multiply that by tanh function before predict/output
  }
  
  call(stateTensor, options = {}) {
    const { batchSize = agent.batch_size } = options;
    //console.log(`stateTensor: ${stateTensor}`);
    // if action bounds not +-1 can multiply here
    // , {batchSize: 4} defaults to 32
    
    const pred = this.model.predict(stateTensor, {batchSize});
    
    return pred;
   
  }
}
//alpha,beta,input_dims,tau,env,gamma,update_actor_interval = 2, warmup = 1000, n_actions=2,max_size=1000000,layer1_size=400,layer2_size=300, batch_size=100,noise=0.1 (video)
//min-max actions is because of noise (video)
// alpha = learning rate for actor (.001), beta = learning rate for critic (.002), tau = target weight update rate (slow is good)
//gamma = discount factor (0 is immediate rewards, 1 is long term rewards and possibly more exploration)
//Phil: batch_size = 300, warmup = 1000, n_games 1000
//n_actions = 2, cInputShape = 9, alpha = 0.001, beta = 0.002, gamma = 0.99, tau = 0.005, warmup = 50, RBufferSize = 100000
class Agent {
  constructor(n_actions = 2, inputShapeA = 47, inputShapeC = 49, alpha = 0.0001, beta = 0.0002, gamma = 0.99, tau = 0.0005, warmup = 128, RBufferSize = 500000) {                           
    this.actor_main = new Actor(n_actions,inputShapeA);
    this.actor_target = new Actor(n_actions,inputShapeA);
    this.critic_main = new Critic(inputShapeC);
    this.critic_main2 = new Critic(inputShapeC);
    this.critic_target = new Critic(inputShapeC);
    this.critic_target2 = new Critic(inputShapeC);
    this.loadedFiles = [];
    this.batch_size = 64;
    this.n_actions = n_actions;
    this.alpha = alpha;
    this.beta = beta;
    this.a_opt = tf.train.adam(alpha);
    this.c_opt1 = tf.train.adam(beta);
    this.c_opt2 = tf.train.adam(beta);
    this.memory = new RBuffer(RBufferSize, [observationSpace.defaults[0].length], this.n_actions); // [stateSpace dimentions]
    this.gamma = gamma;
    this.tau = tau;
    this.actor_update_steps = 2; 
    this.warmup = warmup; // initialy 200
    this.trainstep = 0;
    this.maxStepCount = 256; // not tied to trainstep
    this.decScale = 100000;
    this.min_action = actionSpace.low[0];   // negative movement
    this.max_action = actionSpace.high[0];  // positive movement

    //video sets noise here. and update_network_parameters(tau=1)
    this.actor_main.model.compile({optimizer: this.a_opt, loss: tf.losses.meanSquaredError}); 
    this.actor_target.model.compile({optimizer: this.a_opt, loss: tf.losses.meanSquaredError}); 
    this.critic_main.model.compile({optimizer: this.c_opt1, loss: tf.losses.meanSquaredError});
    this.critic_target.model.compile({optimizer: this.c_opt1, loss: tf.losses.meanSquaredError});
    this.critic_main2.model.compile({optimizer: this.c_opt2, loss: tf.losses.meanSquaredError}); 
    this.critic_target2.model.compile({optimizer: this.c_opt2, loss: tf.losses.meanSquaredError}); 
    
    
    //this.actor_target.model.compile({ optimizer: this.a_opt,loss: tf.losses.meanSquaredError,metrics: ['mse'], }); 
    this.updateTargets(1); // tau = 1 for first update to cause a hard update. target networks gets set to main networks
  }

  act(state, evaluate = false) {
  if (this.trainstep > this.warmup) {
    evaluate = true;
  }
  let returnValue = tf.tidy(() => {     
    /* if self.time_step < self.warmup:   // from Phil's AI learns to walk TD3 video
        mu = np.random.normal(scale=self.noise, size=(self.n_actions,))
      else:
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        mu = self.actor(state)[0] // returns batch size of 1, want scalar
      mu_prime = mu + np.random.normal(scale=self.noise)
      mu_prime = tf.clip_by_value(mu_prime, self.min_action, self.max_action)

      self.time_step += 1

      return mu_prime
    */ 
  
  const stateTensor = tf.tensor(state);
  //console.log(`state tensor: ${stateTensor}, with shape: ${stateTensor.shape}`);
  // STEP 1b: get actions from actor_main
  let actions = this.actor_main.call(stateTensor, {batchSize: 1});
  //console.log(`Actor actions: ${actions}, with shape: ${actions.shape}`);
  //console.log(`reshapedActions: ${reshapedActions}, with shape: ${reshapedActions.shape}`); 
  if (!evaluate) { // meaning it is training
    const noise = tf.randomNormal(actions.shape, 0.0, 0.1); 
    //console.log(`noise: ${noise}, with shape: ${noise.shape}`);
    actions = tf.add(actions, noise);
  }
  if (UI.trainModeCheckbox.checked) {
    const noise = tf.randomNormal(actions.shape, 0.0, 0.5); 
    actions = tf.add(actions, noise);
  }
  //console.log(`act actions: ${actions}`);
  // video adds noise in here. clamping actions prevent noise causing going over or under
  //video sets state, mu, and my_prime
  // we clip actions because noise can cause them to go outside the bounds
  const clippedActions = tf.clipByValue(actions, this.min_action, this.max_action); 
  const scaledActions = tf.mul(clippedActions, tf.scalar(this.max_action));
  //console.log(`scaledA: ${scaledActions}`); 
  // Step 1c: returned action is the first element of a flatened clipped action array
  return scaledActions.arraySync()[0];  //--> [1,-1]
}); // end tidy
return returnValue;
}
  // agent.savexp(state, next_state, action, isDone, reward); 
  savexp(state, next_state, action, done, reward) { // "remember" (video)
    this.memory.storexp(state, next_state, action, done, reward);
  }

  updateTargets(tau = null) { // numbersCopy = JSON.parse(JSON.stringify(nestedNumbers));
    if (tau == null) {
      tau = this.tau;
    }
    
    // STEP 14: we update the weights of the Actor target by polyak averaging
    const weights1 = [];
    
   // const targets1 = this.actor_target.model.getWeights().map(tf.clone); // is clone ok?
    //console.log(`target weights1: ${targets1}`);
    //const mainWeights1 = this.actor_main.model.getWeights().map(tf.clone);
    //const targets1 = this.actor_target.model.getWeights(); // .map(w => w.clone())
    //const mainWeights1 = this.actor_main.model.getWeights();
    const targets1 = this.actor_target.model.getWeights().map(w => w.clone());
    const mainWeights1 = this.actor_main.model.getWeights().map(w => w.clone());
    //console.log(`mainweights1: ${mainWeights1}`);
    for (let i = 0; i < mainWeights1.length; i++) {
      const updatedWeight1 = tf.tidy(() => {
        const weightedMain1 = tf.mul(mainWeights1[i], tf.scalar(tau));
        //console.log(`weigthedMain ${weightedMain1}`);
        const weightedTarget1 = tf.mul(targets1[i], tf.scalar(1 - tau)); // scalar tau only?
        //console.log(`weightedTarget ${weightedTarget1}`);
        const combinedWeight1 = tf.add(weightedMain1, weightedTarget1);
        //console.log(`combinedWeight ${combinedWeight}`);
        return combinedWeight1;
      }); // end tidy
      //console.log(`updatedWeight ${updatedWeight}`);
      weights1.push(updatedWeight1);
      //targets1[i].assign(updatedWeight1);
      
    } ;// end loop 1
    //console.log(`actor_target weights ${this.actor_main.model.getWeights()}`);
    this.actor_target.model.setWeights(weights1);
    //console.log(`updated target model: ${this.actor_target.model.getWeights()}`);
    // STEP 15: we update the weights of the Critic targets by polyak averaging
    const weights2 = [];
    const targets2 = this.critic_target.model.getWeights().map(w => w.clone());
    const mainWeights2 = this.critic_main.model.getWeights().map(w => w.clone());
    //const targets2 = this.critic_target.model.getWeights();
    //const mainWeights2 = this.critic_main.model.getWeights();
    for (let i = 0; i < mainWeights2.length; i++) {
      const updatedWeight2 = tf.tidy(() => {
        const weightedMain2 = tf.mul(mainWeights2[i], tf.scalar(tau));
        const weightedTarget2 = tf.mul(targets2[i], tf.scalar(1 - tau));
        const combinedWeight2 = tf.add(weightedMain2, weightedTarget2);
        return combinedWeight2;
      }); // end tidy
      weights2.push(updatedWeight2);
      //targets2[i].assign(updatedWeight2);
    } // end loop

    this.critic_target.model.setWeights(weights2);
   
    const weights3 = [];
    const targets3 = this.critic_target2.model.getWeights().map(w => w.clone());
    const mainWeights3 = this.critic_main2.model.getWeights().map(w => w.clone());
    //const targets3 = this.critic_target2.model.getWeights();
    //const mainWeights3 = this.critic_main2.model.getWeights();
    for (let i = 0; i < mainWeights3.length; i++) {
      const updatedWeight3 = tf.tidy(() => {
        const weightedMain3 = tf.mul(mainWeights3[i], tf.scalar(tau));
        const weightedTarget3 = tf.mul(targets3[i], tf.scalar(1 - tau));
        const combinedWeight3 = tf.add(weightedMain3, weightedTarget3);
        return combinedWeight3;
      }); // end tidy
      weights3.push(updatedWeight3);
      //targets3[i].assign(updatedWeight3);
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
      //console.log(nextStates);
    //console.log(`states:${states}, nextStates:${nextStates}, rewards:${rewards}, actions:${actions}, dones:${dones}`); // BIG ARRAY WARNING
    // video has all these using critic_1, says doesnt matter. dtype matters
    const statesTensor = tf.tensor(states).squeeze();
    const nextStatesTensor = tf.tensor(nextStates).squeeze();
    const rewardsTensor = tf.tensor(rewards); //[]
    const actionsTensor = tf.tensor(actions).squeeze(); 
    const donesTensor = tf.tensor(dones) // []
    //console.log(`states:${statesTensor}, nextStates:${nextStatesTensor}, rewards${rewardsTensor}, actions:${actionsTensor}, dones:${donesTensor}`);
    //console.log(`nextStatesTensor: ${nextStatesTensor}, shape: ${nextStatesTensor.shape}`);
    //console.log(`donesTensor: ${donesTensor}`);
    //console.log(`rewardsTensor: ${rewardsTensor}`);
     
    // video zeros the optimizers here?

      // STEP 10a: the two critic models take each the couple (s, a) as input and return Q-values as outputs: Q1(s, a), Q2(s, a)
      // STEP 11: we compute the loss coming from the two Critic models: criticLoss = MSE_Loss(Q1(s, a), Qt) + MSE_Loss(Q2(s, a), Qt)
      // Video 2 adds the loss functions together
    function lossFunction1() { 
      const targetActions1 = agent.actor_target.call(nextStatesTensor, {batchSize: agent.batch_size}); 
      const nextStates1 = tf.clone(nextStatesTensor); 
      const rewards1 = tf.clone(rewardsTensor);
      const dones1 = tf.clone(donesTensor);
      const states1 = tf.clone(statesTensor);
      const actions1 = tf.clone(actionsTensor);

      //console.log(`targetActions 1: ${targetActions1}`);
      //console.log(`rewards1: ${rewards1}`);
      // STEP 6: We add Gaussian noise to the next action a` and we clamp it in a range of values supported by environment
      const ranNormalActions1 = tf.add(targetActions1, tf.clipByValue(tf.randomNormal(targetActions1.shape, 0.0, 0.2), -0.5, 0.5));//.map(tf.clone); // video 2 uses actionsTensor?
      //console.log(`randomNormal actions: ${ranNormalActions1}, ${ranNormalActions1.shape}`);
      const clipActions1 = tf.mul(tf.scalar(agent.max_action), tf.clipByValue(ranNormalActions1, agent.min_action, agent.max_action));//.map(tf.clone); 
      //console.log(`cliped actions 1: ${clipActions1}`);
      // STEP 7: The two Critic targets take each the couple (s`, a`) as input and return two Q-values as outputs
      const targetNextStateValues1_1 = agent.critic_target.call(nextStates1, clipActions1).squeeze([1]);
      const targetNextStateValues1_2 = agent.critic_target2.call(nextStates1, clipActions1).squeeze([1]);
      // Shape is [batch_size, 1], want to collaps to [batch_size]. (squeeze)
      // STEP 8: we keep the minimum of the two Q-values. min(Qt1, Qt2)
      const nextTargetStateValue1 = tf.minimum(targetNextStateValues1_1, targetNextStateValues1_2);
      //console.log(`NTSV: ${nextTargetStateValue1}, ${nextTargetStateValue1.shape}`);
      // STEP 9: we get the final target of the two Critic models (Qt = r + gamma * min(Qt1, Qt2) * dones, where gamma is the discount factor)
      //const targetValues1 = rewards1.add(nextTargetStateValue1.mul(tf.scalar(agent.gamma).mul(donesTensor))); //Phil does 1-dones,mines at save
      const targetValues1 = tf.add(rewards1, tf.mul(tf.scalar(agent.gamma), tf.mul(nextTargetStateValue1, dones1)));
      //console.log(`targetValues: ${targetValues1}, ${targetValues1.shape}`);
      const criticValue1 = agent.critic_main.call(states1, actions1).squeeze([1]); 
      //console.log(`criticValue: ${criticValue1}, ${criticValue1.shape}`);
      const criticLoss1 = tf.losses.meanSquaredError(criticValue1, targetValues1);
      
      //console.log(`Crit Loss 1: ${criticLoss1}`); // , ${criticLoss1.shape}
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

    const criticValue2 = agent.critic_main2.call(states2, actions2).squeeze([1]) // squeese? says wont learn otherwise?
    const criticLoss2 = tf.losses.meanSquaredError(criticValue2, targetValues2);
    //console.log(`Crit Loss 2: ${criticLoss2}`);
    if (utilsAI.doneFlag) {utilsAI.criticLoss2 = criticLoss2.array();}
    return criticLoss2
    };

    // STEP 12: we backpropigate the Critic loss and update the parameters of the critic models through optiizers
    const gWeights1 = this.critic_main.model.getWeights(true);
    //console.log(`gWeights1: ${gWeights1}`);
   
    this.critic_main.model.optimizer.minimize(lossFunction1,gWeights1);

    const gWeights2 = this.critic_main2.model.getWeights(true);
  
    this.critic_main2.model.optimizer.minimize(lossFunction2,gWeights2);
    
    this.trainstep += 1;

    // if self.learn_step_cntr % self.update_actor_iter != 0: return (is oposite)
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

       // updating weights every two iterations here because the original paper says to?
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
      //await dlGroup1();
      //await dlGroup2();
      
      
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
      maxsize: JSON.parse(JSON.stringify(agent.memory.maxsize)),  // integer
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
  /*
  if (UI.randZomCheckbox.checked) {
    let zomLoc = Game.getZombieSpawn(); // gives center
    os.zomCoords[0] = zomLoc[0];
    os.zomCoords[1] = zomLoc[1];
    //os.defaults[0][os.locNums.zX] = parseFloat((zomLoc[0] * os.xyNorm).toFixed(8)); // normalize
    //os.defaults[0][os.locNums.zY] = parseFloat((zomLoc[1] * os.xyNorm).toFixed(8));

    zom1.x = zomLoc[0] - (zom1.width/2);
    zom1.y = zomLoc[1] - (zom1.height/2);
    
  }
  */
 
  if (UI.randCivCheckbox.checked) {
    let civLoc = Game.getCivilianSpawn(); // gives center
    os.civCoords[0] = civLoc[0];
    os.civCoords[1] = civLoc[1];
    //os.defaults[0][os.locNums.zX] = parseFloat((civLoc[0] * os.xyNorm).toFixed(8)); // normalize
    //os.defaults[0][os.locNums.zY] = parseFloat((civLoc[1] * os.xyNorm).toFixed(8));

    civ1.x = civLoc[0] - (civ1.width/2);
    civ1.y = civLoc[1] - (civ1.height/2);
    
  }
  
  if (UI.randAgentCheckbox.checked) {
    //let center = [os.zomCoords[0], os.zomCoords[1]];
    let center = [os.civCoords[0], os.civCoords[1]];
    let agentLoc = Game.getAgentSpawn(center,"civilian"); // gives center
    os.agentCoords[0] = agentLoc[0];
    os.agentCoords[1] = agentLoc[1];
    //os.defaults[0][os.locNums.aX] = parseFloat((agentLoc[0] * os.xyNorm).toFixed(4));
    //os.defaults[0][os.locNums.aY] = parseFloat((agentLoc[1] * os.xyNorm).toFixed(4));
    //const scale = 100000;
    const baseX = agentLoc[0] * os.xyNorm;
    const baseY = agentLoc[1] * os.xyNorm;
    const floorX = Math.floor(baseX * agent.decScale);
    const floorY = Math.floor(baseY * agent.decScale);

    os.defaults[0][os.locNums.aX] = (floorX / agent.decScale);
    os.defaults[0][os.locNums.aY] = (floorY / agent.decScale);

    player.x = agentLoc[0] - (player.width/2);
    player.y = agentLoc[1] - (player.height/2);
  }

  const civDrawX = os.civCoords[0] - (civ1.width/2); // civCoords is center. These are used for drawing
  const civDrawY = os.civCoords[1] - (civ1.height/2);
  Game.civilianMoves.push([civDrawX, civDrawY]);

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
  //const decScale = 100000
  //const stateCoords = JSON.parse(JSON.stringify([OS.agentCoords[0], OS.agentCoords[1]])); // for heading
  //let closeWall = false;
  let hitWall = false;
  //locNums: {aX: 0, aY: 1, zX: 2, zY: 3, dist: 4, angle: 5, sensor1-24: 6-29}
  
   const x = player.speed * actionClone[0]; 
   const y = player.speed * actionClone[1];
   //console.log(`x: ${x}, y: ${y}`);
   //console.log(`osX: ${os[0][0]}, osY: ${os[0][1]}`);

    //locNums: {aX: 0, aY: 1,zX: 2,zY: 3,dist: 4, angle: 5}
   //[agentX, agentY, zomX, zomY, distance, angle]
  
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

  //const zomAction = zom1.chase(); // [x,y]
  //OS.zomCoords[0] = zomAction[0];
  //OS.zomCoords[1] = zomAction[1];
  //console.log(`zomAction: ${zomAction}`);

  //zom1.createPolygon(); // can return the polygon points
  civ1.createPolygon();
  
  player.sensor.update(Game.mapBorders,[civ1]); // hardcoded
  const offsets = player.sensor.readings.map((reading) => {
    if (reading == null) {
        return 1;  // Default value for no detection
    } else {
        // Adjust the offset based on the type of the detected object
        if (reading.type === "civilian") {
            // Map distances from 1 to 0 (decending)
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
  //let highestOffset = 0;
  let detect = false;

  for (let i = 0; i < offsets.length; i++){
  let copy = JSON.parse(JSON.stringify(offsets[i]));
  const clipCopy = parseFloat((copy).toFixed(4));
  if (clipCopy > 0 && clipCopy < 1) {detect = true}
  //highestOffset = Math.max(highestOffset,clipCopy);
  //if (clipCopy < 0.5 && clipCopy > 0 && clipCopy < osNS[0][i+2]) {closeWall = true;}
  osNS[0][i+2] = clipCopy;
  
  }

  let isDone = false;
  let reward = 0;
  let penalty = 0;
  // range [-2, 2]
  
  let civDist = utilsAI.distance(OS.agentCoords[0], OS.agentCoords[1], OS.civCoords[0], OS.civCoords[1]);
  let civAngle = utilsAI.angle([OS.agentCoords[0], OS.agentCoords[1]], [OS.civCoords[0], OS.civCoords[1]]);
  //const zomDist = utilsAI.distance(OS.agentCoords[0], OS.agentCoords[1], OS.zomCoords[0], OS.zomCoords[1]);
  //let zomAngle = utilsAI.angle([OS.agentCoords[0], OS.agentCoords[1]], [OS.zomCoords[0], OS.zomCoords[1]]);
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
  //anglePenalty /= 2; // [0, 0.3] // Share space with distancePenalty
  //anglePenalty *= 2;

  let distancePenalty = civDist;
  let distanceBonus = 1 / civDist;

  let distanceScaling = observationSpace.xyNorm; // [0, 1] xyNorm is hardcoded 0.002 based on map size

  distancePenalty *= distanceScaling;
  distanceBonus *= distanceScaling;

  distancePenalty = Math.min(distancePenalty, 1); // safegards
  distanceBonus = Math.min(distanceBonus, 1);
  //distancePenalty /= 2; // [0, 0.5] // share space with anglePentalty
  //distancePenalty *= 2
 
function calculateTimePenalty(step, maxSteps) {
  const maxPenalty = -0.2; // Adjust as needed
  const timeRatio = step / maxSteps;
  const timePenalty = Math.max(maxPenalty, maxPenalty * timeRatio);
  return timePenalty;
}

  /*
  if (highestOffset > 0) {
    if (civDist < OS.rawDist) {reward += highestOffset;}
    //else {penalty += (1 - highestOffset)}
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
  //osNS[0][LN.dist] = parseFloat((zomDist * OS.xyNorm).toFixed(3));
  //osNS[0][LN.angle] = parseFloat((zomAngle / Math.PI).toFixed(8));
  //osNS[0][LN.angle] = (zomAngle + Math.PI) / (2 * Math.PI);

  const timePenalty = calculateTimePenalty(currentStep, agent.maxStepCount);
  //const timeBonus = calculateTimeBonus(currentStep, agent.maxStepCount);
  //reward += timeBonus;
  
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
  if (civDist <= player.width) { // finding target overules everything.
    reward = 2; 
    isDone = true;
    if (!Game.testing){
      console.warn("AGENT FOUND CIVILIAN!");
      Game.agentWins++;
      UI.agentWins.innerHTML = `Times Won: ${Game.agentWins}`;
      Game.winsThisRun++;
      UI.winsThisRun.innerHTML = `Wins This Run: ${Game.winsThisRun}`;
    }
    else {
      console.info("AGENT FOUND CIVILIAN");
      Game.winsThisRun++;
      UI.winsThisRun.innerHTML = `Wins This Run: ${Game.winsThisRun}`;
    }
  }
  else if (currentStep >= agent.maxStepCount) {
    //reward = 1; // causing location sticking?
     // If the zombie hasn't reached the player then the agent survives, max point
     isDone = true;
      
    }
  
    //console.log(`Step: ${currentStep}`);
    //console.log(`Angle penalty: ${anglePenalty}`);
    //console.log(`Distance penalty: ${distancePenalty}`);
    //console.log(`Time penalty: ${timePenalty}`);
    //console.log(`Final penalty: ${penalty}`);
    //console.log(`Step: ${currentStep}, reward: ${reward}`);
  
  
  const next_State = JSON.parse(JSON.stringify(observationSpace.next_stateSpace));
  //console.log(next_State);
  let terminalFlag = false;
  if (isDone) {terminalFlag = true;}
  const aX = OS.agentCoords[0];
  const aY = OS.agentCoords[1];
  //const zX = OS.zomCoords[0];
  //const zY = OS.zomCoords[1];
  Game.agentMoves.push([aX, aY, terminalFlag]); // For drawing agent paths
  //Game.zombieMoves.push([zX, zY]); // For drawing zombie paths
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
    if (s == episodes-1) {utilsAI.doneFlag = true;}
    // gamma decay
    //if (agent.gamma > 0.2 && episodes % 10 === 0) {
     // agent.gamma -= 0.001
    //}
    // STEP 0: reset everything
    let totalReward = 0;
    envReset();
    let done = false;
    let currentStep = 0;
    let stepDone = false;
    
    while (!done) {
      if (!Game.running) {break}
      //if (utilsAI.doneFlag) {utilsAI.doneFlag = false}
      //console.log(currentStep);
      //console.log(state);
      // STEP 1a: get an action based on the current state 
      //const action = agent.act(state); // choose_action(observation), is envReset(). // video 2 adds more noise to the action
      let state = JSON.parse(JSON.stringify(observationSpace.stateSpace));
      const action = agent.act(state);
      //console.log(action); //  [0,0]
      //const scale = 100000 // 0.1234
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
      
      //const x = parseFloat(action[0].toFixed(4)); 
      //const y = parseFloat(action[1].toFixed(4));
      const clipAction = [x,y];
      //console.log(clipAction);
      // STEP 2a: step the environment with the action, returning the new state, rewards, and if done
      const { next_state, reward, isDone } = envStep(clipAction, currentStep);
      // STEP 3: save the new state to the memory buffer
      //console.log(`env state: ${state}`);
      //console.log(`env next_state: ${next_state}`);
      //console.log(`action after env: ${action}`);
      //console.log(`saving isDone as: ${isDone}`);
      //console.log(`env reward: ${reward}`);
      
      if (!Game.testing){ 
        agent.savexp(state, next_state, clipAction, isDone, reward); 
        // Step 4-15..: train the system
        agent.train(); // only gets called if memory.cnt = agent.batch size
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
