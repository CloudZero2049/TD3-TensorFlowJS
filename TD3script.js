// tf.memory.numTensors() // reduce t o1 round to find where memory leaks are.
//tidy() -->
// const result = tf.scalar(121);
// res1 = tf.keep(result.sqrt());

// Episodes Until maximization: about 20-40x
// High score: 49.86120644246673
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
      //let normalizedAngle = (angle + Math.PI) / (2 * Math.PI); // Normalize to [0, 1]
      //normalizedAngle = normalizedAngle * 2 - 1; // normalise to [-1, 1]
      return angle;
     
    
  }
}

//observationSpace.stateSpace.length
const observationSpace ={ 
  //[agentX,agentY, directionX, directionY, agentSpeedX ,agentSpeedY,civilianX,civilianY,dist to civ, angle to civ] 10 total
  initUpdate: function() { // [0,0],[0,0]
    let defs = this.defaults;
    const aSliderX = document.getElementById("agentRangeX");
    const aSliderY = document.getElementById("agentRangeY");
    const cSliderX = document.getElementById("civRangeX");
    const cSliderY = document.getElementById("civRangeY");
  
    defs[0][0] = parseInt(aSliderX.value) + (player.width/2); // Do I need the centers here?
    defs[0][1] = parseInt(aSliderY.value) + (player.height/2);
    defs[0][6] = parseInt(cSliderX.value) + (civ1.width/2);
    defs[0][7] = parseInt(cSliderY.value) + (civ1.height/2);
    defs[0][8] = utilsAI.distance(defs[0][0], defs[0][1], defs[0][4], defs[0][5]);
    defs[0][9] = utilsAI.angle([defs[0][0],defs[0][1]],[defs[0][4], defs[0][5]]);

  },
  locNums: {
    aX: 0,
    aY: 1,
    dX: 2,
    dY: 3,
    sX: 4,
    sY: 5,
    cX: 6,
    cY: 7,
    dist: 8,
    angle: 9,
  },
  //civLoc:[civ1.x + (civ1.width/2), civ1.y + (civ1.height/2)],
  defaults: [[player.x + (player.width/2), player.y + (player.height/2), 0, 0, 0.1, 0.1, civ1.x + (civ1.width/2), civ1.y + (civ1.height/2), 100, 1]],
  stateSpace: [[player.x + (player.width/2), player.y + (player.height/2), 0, 0, 0.1, 0.1, civ1.x + (civ1.width/2), civ1.y + (civ1.height/2), 100, 1 ]],
  next_stateSpace: [[player.x + (player.width/2), player.y + (player.height/2), 0, 0, 0.1, 0.1, civ1.x + (civ1.width/2), civ1.y + (civ1.height/2), 100, 1]],
}
// [moveX, moveY, speedX, speedY, distance, angle]
const actionSpace = {
  numberActions: 4,
  shape: [4], // not used
  actions: [0,0,0,0], // not used
  low: [-1,-1,-1,-1],
  high: [1,1,1,1]
};

// Can probably set all these to just empty arrays. tf.zeros makes tensors. should be tf.zeros([1, n_actions])? what is the shape?
                          //tf.zeros(shape, dataType)

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
    this.done_memory[index] = JSON.parse(JSON.stringify(1-done)); // 1-done here because array math later. Number(done)
    this.cnt += 1;
  }

  sample(batch_size) {
    const max_mem = Math.min(this.cnt, this.maxsize);
   // let batch; // batch = np.random.choice(max_mem, batch_size) // random choice between 0-max_mem, SHAPE IS BATCH_SIZE?!

    // Randomly sample indices without replacement
    let batch = [];
    for (let i = 0; i < batch_size; i++) {
        let index;
        do {
            index = Math.floor(Math.random() * max_mem);
        } while (batch.includes(index));
        batch.push(index);
    }

    //batch = tf.tensor1d(batch, 'int32');
    //console.log(`batch: ${batch}`);

    const states = batch.map(index => this.state_memory[index]);
    const next_states = batch.map(index => this.next_state_memory[index]);
    const rewards = batch.map(index => this.reward_memory[index]);
    const actions = batch.map(index => this.action_memory[index]);
    const dones = batch.map(index => this.done_memory[index]);

    //console.log(next_states);
   //const states = tf.gather(this.state_memory, batch);
   //const next_states = tf.gather(this.next_state_memory, batch);
  // const rewards = tf.gather(this.reward_memory, batch);
   //const actions = tf.gather(this.action_memory, batch);
   //const dones = tf.gather(this.done_memory, batch);
   // batch = Math.floor(Math.random() * max_mem);
    
    // tf.gather(x, indices, axis?, batchDims?) x = tensor
    //const states = JSON.parse(JSON.stringify(this.state_memory[batch]));
    //const next_states = JSON.parse(JSON.stringify(this.next_state_memory[batch]));
   // const rewards = JSON.parse(JSON.stringify(this.reward_memory[batch]));
    //const actions = JSON.parse(JSON.stringify(this.action_memory[batch]));
   // const dones = JSON.parse(JSON.stringify(this.done_memory[batch]));
    
    return [states, next_states, rewards, actions, dones];
  }
}

class Critic {  // it is possible to combine the 4 critics into 2 critics supposedly
  constructor(inputShape) { // completely different in video. checkpoints models to files here
    this.model = tf.sequential();
    this.model.add(tf.layers.dense({ inputShape: [inputShape], units: 512, activation: 'relu', dtype: 'float32', kernelInitializer: 'randomNormal'})); // state_dim + action_dim (video 2)
    this.model.add(tf.layers.dense({ units: 512, activation: 'relu', dtype: 'float32', kernelInitializer: 'randomNormal' }));
    this.model.add(tf.layers.dense({ units: 1, activation: null, dtype: 'float32', kernelInitializer: 'randomNormal' })); // Output to 1 Q value
  }

  call(inputstate, action) { // feed forward
    //console.log(`inputstate: ${inputstate}`);
    //console.log(`action: ${action}`);
   
   // STEP 10b: we concatinate the input state[6] with action
   // Repeat the observation state tensor for each example in the batch
  //const batchSize = action.shape[0];
  //console.log(`action: ${action.shape}`);
  //const repeatedObservationStateTensor = inputstate.tile([batchSize, 1]);
  //console.log(`rep-OST: ${inputstate.shape}`);
  // Concatenate the two tensors along the last axis (axis 1)
    const catTensor = tf.concat([inputstate, action], 1);
    //let oldCat = inputstate.concat(action); doesn't work, says shape is different
   //const normalCat = normalizeData(catTensor);
   //console.log(`catTensor: ${catTensor.shape}`);
    const x = this.model.predict(catTensor);
    
    return x;
  
  }
}
//kernelInitializer: tf.randomNormal(shape)
class Actor {
  constructor(n_actions,inputShape) { // has alpha in constructor (video)
    this.model = tf.sequential();
    this.model.add(tf.layers.dense({ inputShape: inputShape, units: 512, activation: 'relu', dtype: 'float32', kernelInitializer: 'randomNormal'  })); // input state_dim (video 2)
    this.model.add(tf.layers.dense({ units: 512, activation: 'relu', dtype: 'float32', kernelInitializer: 'randomNormal'   }));
    this.model.add(tf.layers.dense({ units: n_actions, activation: 'tanh', dtype: 'float32', kernelInitializer: 'randomNormal'  })); // output units: is number of actions/action space
    // from DDPG paper. pi - tangent hyperbolic, +-1
    // if action bounds are say +-2, multiply that by tanh function before predict/output
  }
  
  call(stateTensor) {
    //console.log(`stateTensor: ${stateTensor}`);
    // if action bounds not +-1 can multiply here
    //const normalState = normalizeData(stateTensor);
    const pred = this.model.predict(stateTensor);
    //console.log(`pred: ${pred}`);
    return pred;
   
  }
}
//alpha,beta,input_dims,tau,env,gamma,update_actor_interval = 2, warmup = 1000, n_actions=2,max_size=1000000,layer1_size=400,layer2_size=300, batch_size=100,noise=0.1 (video)
//min-max actions is because of noise (video)
//warmups. 32,64,128,256,512
// alpha = learning rate for actor (.001), beta = learning rate for critic (.002), tau = target weight update rate (slow is good)
//gamma = discount factor (0 is immediate rewards, 1 is long term rewards and possibly more exploration)
//Phil: batch_size = 300, warmup = 1000, n_games 1000
//n_actions = 2, cInputShape = 9, alpha = 0.001, beta = 0.002, gamma = 0.99, tau = 0.005, warmup = 50, RBufferSize = 100000
class Agent {
  constructor(n_actions = 4, inputShapeA = 8, inputShapeC = 10, alpha = 0.0001, beta = 0.002, gamma = 0.90, tau = 0.005, warmup = 64, RBufferSize = 100000) {                           
    this.actor_main = new Actor(n_actions,inputShapeA);
    this.actor_target = new Actor(n_actions,inputShapeA);
    this.critic_main = new Critic(inputShapeC);
    this.critic_main2 = new Critic(inputShapeC);
    this.critic_target = new Critic(inputShapeC);
    this.critic_target2 = new Critic(inputShapeC);
    this.loadedFiles = [];
    this.batch_size = 64;
    this.n_actions = n_actions;
    this.a_opt = tf.train.adam(alpha);
    this.c_opt1 = tf.train.adam(beta);
    this.c_opt2 = tf.train.adam(beta);
    this.memory = new RBuffer(RBufferSize, [observationSpace.stateSpace[0].length], this.n_actions); // [stateSpace dimentions]
    this.gamma = gamma;
    this.tau = tau;
    this.actor_update_steps = 2; 
    this.warmup = warmup; // initialy 200
    this.trainstep = 0;
    this.maxStepCount = 64; // not tied to trainstep
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
    this.updateTarget(1); // tau = 1 for first update to cause a hard update. target networks gets set to main networks
  }

  act(state, evaluate = false) { // observation
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
  
  const stateTensor = tf.tensor(state); // [] added again because of DDPG video, then again because CGPT
  //console.log(`state tensor: ${stateTensor}, with shape: ${stateTensor.shape}`);
  // STEP 1b: get actions from actor_main
  let actions = this.actor_main.call(stateTensor);
  //let reshapedActions = tf.reshape(tf.clone(actions),[2]);
  //if (!actions.shape) {actions = tf.reshape(actions,[1,4]);}
  //console.log(`Actor actions: ${actions}, with shape: ${actions.shape}`);
  //console.log(actions);
  //console.log(`reshapedActions: ${reshapedActions}, with shape: ${reshapedActions.shape}`); 
  if (!evaluate) { // meaning it is training
    const noise = tf.randomNormal(actions.shape, 0.0, 0.1); // video uses mu, adds noise. need more noise?
    //console.log(`noise: ${noise}, with shape: ${noise.shape}`);
    //actions.add(noise);
    actions = tf.add(actions, noise);
    //console.log(`after noise: ${actions}, with shape: ${actions.shape}`);
  }
  //console.log(`act actions: ${actions}`);
  // video adds noise in here. clamping actions prevent noise causing going over or under
  //video sets state, mu, and my_prime
  // we clip actions because noise can cause them to go outside the bounds
  const clippedActions = tf.clipByValue(actions, this.min_action, this.max_action); 
  //const scaledActions = tf.mul(clippedActions, maxTensor); // mac_action isn't tensor, but is 1 anyway
  const scaledActions = tf.mul(clippedActions, tf.scalar(this.max_action));
  //console.log(`scaledA: ${scaledActions}`); 
  // self.time_step += 1 (video)
  // Step 1c: returned action is the first element of a flatened clipped action array
  return scaledActions.arraySync()[0];  //--> [1,-1]
}); // end tidy
return returnValue;
}
  // agent.savexp(state, next_state, action, isDone, reward); 
  savexp(state, next_state, action, done, reward) { // "remember" (video)
    this.memory.storexp(state, next_state, action, done, reward);
  }

  updateTarget(tau = null) { // numbersCopy = JSON.parse(JSON.stringify(nestedNumbers));
    if (tau == null) {
      tau = this.tau;
    }
    
    // STEP 14: we update the weights of the Actor target by polyak averaging
    const weights1 = [];
    //const targets1 = JSON.parse(JSON.stringify(this.actor_target.model.getWeights()));
    //const targets1 = this.actor_target.model.getWeights();
    const targets1 = this.actor_target.model.getWeights().map(tf.clone);
    //console.log(`target weights1: ${targets1}`);
    const mainWeights1 = this.actor_main.model.getWeights().map(tf.clone);
    //weights1.append ( weight * tau + targets1[i]*(1-tau))
    //console.log(`mainweights1: ${mainWeights1}`);
    for (let i = 0; i < mainWeights1.length; i++) {
      const updatedWeight1 = tf.tidy(() => {
        const weightedMain1 = tf.mul(mainWeights1[i], tf.scalar(tau));
        //console.log(`weigthedMain ${weightedMain1}`);
        const weightedTarget1 = tf.mul(targets1[i], tf.scalar(1 - tau));
        //console.log(`weightedTarget ${weightedTarget1}`);
        const combinedWeight1 = tf.add(weightedMain1, weightedTarget1);
        //console.log(`combinedWeight ${combinedWeight}`);
        return combinedWeight1;
      }); // end tidy
      //console.log(`updatedWeight ${updatedWeight}`);
      weights1.push(updatedWeight1);
      
    } ;// end loop 1
    this.actor_target.model.setWeights(weights1);
    //console.log(`updated target model: ${this.actor_target.model.getWeights()}`);
    // STEP 15: we update the weights of the Critic targets by polyak averaging
    const weights2 = [];
    const targets2 = this.critic_target.model.getWeights().map(tf.clone);
    const mainWeights2 = this.critic_main.model.getWeights().map(tf.clone);
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
    const targets3 = this.critic_target2.model.getWeights().map(tf.clone);
    const mainWeights3 = this.critic_main2.model.getWeights().map(tf.clone);
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
    
    if (this.memory.cnt < this.batch_size) { // important, sample will fail othewise
        return; 
    }
    tf.tidy(() => {
    // STEP 4: we sample a batch of transitions (s, s`, a, r) from memory
    const [states, nextStates, rewards, actions, dones] = this.memory.sample(this.batch_size);
      
    //console.log(`states:${states}, nextStates:${nextStates}, rewards:${rewards}, actions:${actions}, dones:${dones}`); // BIG ARRAY WARNING
    // video has all these using critic_1, says doesnt matter. dtype matters
    const statesTensor = tf.tensor(states).squeeze();
    const nextStatesTensor = tf.tensor(nextStates).squeeze();
    const rewardsTensor = tf.tensor(rewards); //[]
    const actionsTensor = tf.tensor(actions).squeeze(); 
    const donesTensor = tf.tensor(dones) // []
    //console.log(`states:${statesTensor}, nextStates:${nextStatesTensor}, rewards${rewardsTensor}, actions:${actionsTensor}, dones:${donesTensor}`);
    //console.log(`statesTensor:${statesTensor.shape}, nextStatesTensor:${nextStatesTensor.shape}, rewardsTensor${rewardsTensor.shape}, actionsTensor:${actionsTensor.shape}, donesTensor:${donesTensor.shape}`);
    //console.log(`nextStatesTensor: ${nextStatesTensor}, shape: ${nextStatesTensor.shape}`);
    //console.log(`donesTensor: ${donesTensor}`);
    //console.log(`rewardsTensor: ${rewardsTensor}`);
     //targetActions = targetActions.mul(this.max_action).clipByValue(this.min_action, this.max_action);
    // from video. when new states are terminal (0s). where done is true, values are set to 0
    //q1_[done] = 0.0 
    //q2_[done] = 0.0
    //dimentional manipulation
    //q1_ = q1_view(-1)
    //q2_ = q2_.view(-1)
    //target = reward + self.gamma*critic_value_  (video)
    //target = target.view(self.batch_size, 1) (video) doesnt have .mul(dones)
    // video zeros the optimizers here

      // STEP 10a: the two critic models take each the couple (s, a) as input and return Q-values as outputs: Q1(s, a), Q2(s, a)
      // STEP 11: we compute the loss coming from the two Critic models: criticLoss = MSE_Loss(Q1(s, a), Qt) + MSE_Loss(Q2(s, a), Qt)
      // Video 2 adds the loss functions together
    function lossFunction1() { 
      const targetActions1 = agent.actor_target.call(nextStatesTensor); 
      const rewards1 = tf.clone(rewardsTensor);
      
      //console.log(`targetActions actions: ${targetActions1}`);
      //console.log(`rewards1: ${rewards1}`);
      //console.log(`dones: ${donesTensor}`);
      // STEP 6: We add Gaussian noise to the next action a` and we clamp it in a range of values supported by environment
      const ranNormalActions1 = tf.add(targetActions1, tf.clipByValue(tf.randomNormal(targetActions1.shape, 0.0, 0.2), -0.5, 0.5));//.map(tf.clone); // video 2 uses actionsTensor?
      //console.log(`randomNormal actions: ${ranNormalActions1}, ${ranNormalActions1.shape}`);
      const clipActions1 = tf.mul(tf.scalar(agent.max_action), tf.clipByValue(ranNormalActions1, agent.min_action, agent.max_action));//.map(tf.clone); 
      //console.log(`cliped actions: ${clipActions1}, ${clipActions1.shape}`);
      // STEP 7: The two Critic targets take each the couple (s`, a`) as input and return two Q-values as outputs
      const targetNextStateValues1_1 = agent.critic_target.call(nextStatesTensor, clipActions1).squeeze([1]);
      const targetNextStateValues1_2 = agent.critic_target2.call(nextStatesTensor, clipActions1).squeeze([1]);
      // Shape is [batch_size, 1], want to collaps to [batch_size]. (squeeze)
      // STEP 8: we keep the minimum of the two Q-values. min(Qt1, Qt2)
      const nextTargetStateValue1 = tf.minimum(targetNextStateValues1_1, targetNextStateValues1_2);
      //console.log(`NTSV: ${nextTargetStateValue1}, ${nextTargetStateValue1.shape}`);
      //console.log(`Scaled NTSV: ${tf.mul(tf.scalar(agent.gamma), tf.mul(nextTargetStateValue1, donesTensor))}`);
      // STEP 9: we get the final target of the two Critic models (Qt = r + gamma * min(Qt1, Qt2) * dones, where gamma is the discount factor)
      //const targetValues1 = rewards1.add(nextTargetStateValue1.mul(tf.scalar(agent.gamma).mul(donesTensor))); //Phil does 1-dones,mines at save
      const targetValues1 = tf.add(rewards1, tf.mul(tf.scalar(agent.gamma), tf.mul(nextTargetStateValue1, donesTensor)));
      // Phil says "that will set the value of the second term, gamma*critic value to 0 everyhwere the done flag is true"
      //console.log(`targetValues: ${targetValues1}, ${targetValues1.shape}`);
      // says we have to squeeze because we have batch dimentino, and doesn't learn if you past that through.
      const criticValue1 = agent.critic_main.call(statesTensor, actionsTensor).squeeze([1]); // squeese? says wont learn otherwise
      //console.log(`criticValue: ${criticValue1}, ${criticValue1.shape}`);
      const criticLoss1 = tf.losses.meanSquaredError(criticValue1, targetValues1);
      //console.log(`crit Loss1: ${criticLoss1}`); // , ${criticLoss1.shape}
     
      return criticLoss1;
    };

    function lossFunction2() { 
      const targetActions2 = agent.actor_target.call(nextStatesTensor); 
      const rewards2 = tf.clone(rewardsTensor);

      const ranNormalActions2 = tf.add(targetActions2, tf.clipByValue(tf.randomNormal(targetActions2.shape, 0.0, 0.2), -0.5, 0.5));
      const clipActions2 = tf.mul(tf.scalar(agent.max_action), tf.clipByValue(ranNormalActions2, agent.min_action, agent.max_action));
    
      const targetNextStateValues2_1 = agent.critic_target.call(nextStatesTensor, clipActions2).squeeze([1]);
      const targetNextStateValues2_2 = agent.critic_target2.call(nextStatesTensor, clipActions2).squeeze([1]);
      const nextTargetStateValue2 = tf.minimum(targetNextStateValues2_1, targetNextStateValues2_2);
      const targetValues2 = tf.add(rewards2, tf.mul(tf.scalar(agent.gamma), tf.mul(nextTargetStateValue2, donesTensor)));

    const criticValue2 = agent.critic_main2.call(statesTensor, actionsTensor).squeeze([1]) // squeese? says wont learn otherwise?
    const criticLoss2 = tf.losses.meanSquaredError(criticValue2, targetValues2);
  
    return criticLoss2
    //return criticMean2;
    };

    
    // STEP 12: we backpropigate the Critic loss and update the parameters of the critic models through optiizers
    const gWeights1 = this.critic_main.model.getWeights(true)//.map(tf.clone); // this does give all the weights right?
    //console.log(`gWeights1: ${gWeights1}`);
    //this.c_opt1.computeGradients(lossFunction1,gWeights1);
    //let computedgrads1 = this.c_opt1.computeGradients(lossFunction1,gWeights1);  // do we need to .step() optimizers?
    //const computedgrads1 = tf.variableGrads(lossFunction1,gWeights1);
    //this.c_opt1.applyGradients(computedgrads1.grads);
    //this.critic_main.model.optimizer.applyGradients(computedgrads1.grads);
    this.critic_main.model.optimizer.minimize(lossFunction1,gWeights1);

    const gWeights2 = this.critic_main2.model.getWeights(true)//.map(tf.clone);
    //let computedgrads2 = this.c_opt2.computeGradients(lossFunction2,gWeights2);
   // const computedgrads2 = tf.variableGrads(lossFunction2,gWeights2);
    this.critic_main2.model.optimizer.minimize(lossFunction2,gWeights2);
    //this.c_opt2.applyGradients(computedgrads2.grads);
    //this.critic_main2.model.optimizer.applyGradients(computedgrads2.grads);
    
    this.trainstep += 1;

    // if self.learn_step_cntr % self.update_actor_iter != 0: return (is oposite)
    // STEP 13: once every two iterations, we update the Actor model by performing gradient ascent on the output of the first critic model
    if (this.trainstep % this.actor_update_steps === 0) {
      
      function lossFunction3() {
        // gradient ascent is the negative of gradient decent.
        //console.log(`statesTensor: ${statesTensor}, ${statesTensor.shape}`);
        const actorCall = agent.actor_main.call(statesTensor);
        //console.log(`actorCall: ${actorCall}, ${actorCall.shape}`);
        const criticCall = agent.critic_main.call(statesTensor, actorCall).neg();
        //console.log(`criticCall: ${criticCall}, ${criticCall.shape}`);
        const actorLoss = tf.mean(criticCall);
        //console.log(`actor loss: ${actorLoss}`);
        return actorLoss;
           // const new_policy_actions = self.actor_main(states)
           //  let actor_loss = -self.critic_main(states, new_policy_actions)
            // let actor_loss = tf.math.reduce_mean(actor_loss)
      }
      
      const gWeights3 = this.actor_main.model.getWeights(true)//.map(tf.clone);
      //let computedgrads3 = this.a_opt.computeGradients(lossFunction3,gWeights3);
      //const computedgrads3 = tf.variableGrads(lossFunction3,gWeights3);
      this.actor_main.model.optimizer.minimize(lossFunction3,gWeights3);
      //console.log(computedgrads3.grads);
      //this.a_opt.applyGradients(computedgrads3.grads);
      //this.actor_main.model.optimizer.applyGradients(computedgrads3.grads);

       // STEP 14/15... updating weights every two iterations. moved here because the original paper says to.
      //this.updateTarget(); // same as self.update_netowrk_parameters() in video
      
    }
    this.updateTarget();
  }); // End Tidy
  }
  
  async downloadModels() { // await model.save('localstorage://demo/management/model1');
    try {
      await agent.actor_main.model.save('downloads://actor_main-model');
      await agent.actor_target.model.save('downloads://actor_target-model');
      await agent.critic_main.model.save('downloads://critic_main-model');
      await agent.critic_target.model.save('downloads://critic_target-model');
      await agent.critic_main2.model.save('downloads://critic_main2-model');
      await agent.critic_target2.model.save('downloads://critic_target2-model');
      
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
      const memoryData = {
      version: Game.version,
      cnt: agent.memory.cnt, // integer
      maxsize: agent.memory.maxsize,  // integer
      state_memory: agent.memory.state_memory,  // tensor // .arraySync()
      next_state_memory: agent.memory.next_state_memory, // tensor
      action_memory: agent.memory.action_memory, // tensor
      reward_memory: agent.memory.reward_memory, // tensor
      done_memory: agent.memory.done_memory // tensor
      }
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
            }
            else {throw`actor_main weights file not found`}
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
          else {throw`actor_target weights file not found`}
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
          else {throw`critic_main weights file not found`}
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
          else {throw`critic_target weights file not found`}
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
          else {throw`critic_main2 weights file not found`}
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
          else {throw`critic_target2 weights file not found`}
        }
        break;     
      }
      UI.modelsLoadedInfo.innerHTML = `Models Loaded: ${agent.loadedFiles}`;
      
    } // end i loop
    } catch (error) {
      console.error(`failed to load model: ${error}`);
    }
   // drag and drop
  // let dropbox;

  // dropbox = document.getElementById("dropbox");
  // dropbox.addEventListener("dragenter", dragenter, false);
  // dropbox.addEventListener("dragover", dragover, false);
  // dropbox.addEventListener("drop", drop, false);
  /*
  function dragenter(e) {
    e.stopPropagation();
    e.preventDefault();
  }
  
  function dragover(e) {
    e.stopPropagation();
    e.preventDefault();
  }
  */
 /*
  function drop(e) {
    e.stopPropagation();
    e.preventDefault();
  
    const dt = e.dataTransfer;
    const files = dt.files;
  
    handleFiles(files);
  }
  */
  }
  loadMemory(file) {
    
    if (file) {
    const reader = new FileReader();
    
    reader.onload = function (e) {
      try {
        const parsedData = JSON.parse(e.target.result);
        if (parsedData.version !== Game.version) {
          let ignorVer = confirm("The memory version and program version don't match. Continue anyway?")
          if (!ignorVer) {return}
        }
        agent.memory.cnt = parsedData.cnt;
        agent.memory.maxsize = parsedData.maxsize;
        agent.memory.state_memory = parsedData.state_memory;//,observationSpace.stateSpace[0].length;
        agent.memory.next_state_memory = parsedData.next_state_memory;//, observationSpace.stateSpace[0].length;
        agent.memory.action_memory = parsedData.action_memory;//, agent.n_actions;
        agent.memory.reward_memory = parsedData.reward_memory;
        agent.memory.done_memory = parsedData.done_memory;
      
        console.log(`Loaded memory succesfully with count: ${parsedData.cnt}`);
        
        
      } catch (error){
        console.error('Error parsing JSON or storing memory:', error);
      }
    };

    reader.readAsText(file);
  }
  }
  async copyModel(from,to) {
    /** Copy the model, from Local Storage to IndexedDB.
    await tf.io.copyModel(
    'localstorage://demo/management/model1',
    'indexeddb://demo/management/model1');
    
    */
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
  // reset the game environment and entities
  // set random spawn
  const os = observationSpace;
  if (UI.randAgentCheckbox.checked) {
    let center = [os.defaults[0][os.locNums.cX], os.defaults[0][os.locNums.cY]];
    let agentLoc = Game.getAgentSpawn(center);
    os.defaults[0][os.locNums.aX] = agentLoc[0];
    os.defaults[0][os.locNums.aY] = agentLoc[1];
  }
  // Reset the state space
  os.stateSpace = JSON.parse(JSON.stringify(os.defaults));
  os.next_stateSpace = JSON.parse(JSON.stringify(os.defaults));
  
  const state = os.stateSpace;
  
  return state;
}



function envStep(action, state, currentStep) {
  const actionClone = JSON.parse(JSON.stringify(action));
  const playerSpeed = 5;  // temp hardcode 
  const os = observationSpace.next_stateSpace;
  const LN = observationSpace.locNums
  let hitWall = false;
  //locNums: {aX: 0,aY: 1,dx: 2,dY: 3,sX: 4,sY: 5,cX: 6,cY: 7,dist: 8, angle: 9,}

  //console.log(action);
   // Calculate new positions of entities, resolve collisions, etc.
   const x = (playerSpeed * actionClone[2]); 
   const y = (playerSpeed * actionClone[3]);
   //console.log(`x: ${x}, y: ${y}`);
   //console.log(`osX: ${os[0][0]}, osY: ${os[0][1]}`);

    //locNums: {aX: 0,aY: 1,dx: 2,dY: 3,sX: 4,sY: 5,cX: 6,cY: 7,dist: 8, angle: 9,}
   //[agentX, agentY, directionX, directionY, speedX, speedY, civX, civY, distance, angle]
  // move X
  const movementThreshold = 0.1;

  if (actionClone[0] < -movementThreshold) { // move left
    if ((os[0][LN.aX] + x) < 10) {os[0][LN.aX] = 10; hitWall = true;}
    else {os[0][LN.aX] += x;}
    os[0][LN.dX] = -1;
  }
  else if (actionClone[0] > movementThreshold) { // move right
    if ((os[0][LN.aX] + x) > (Game.width - 10)) {os[0][LN.aX] = (Game.width - 10); hitWall = true;} // player width & height = 20. x,y is center
    else {os[0][LN.aX] += x;}
    os[0][LN.dX] = 1;
  }
  else {os[0][LN.dX] = 0;} 

  // move Y
  if (actionClone[1] < -movementThreshold) { // move up
    if ((os[0][LN.aY] + y) < 10) {os[0][LN.aY] = 10;  hitWall = true;}
    else {os[0][LN.aY] += y;}
    os[0][LN.dY] = -1;
  }
  else if (actionClone[1] > movementThreshold) { // move down
    if ((os[0][LN.aY] + y) > (Game.height - 10)) {os[0][LN.aY] = (Game.height - 10); hitWall = true;}
    else {os[0][LN.aY] += y;}
    os[0][LN.dY] = 1;
  }
  else {os[0][LN.dY] = 0;}
  //os[0][LN.dX] = actionClone[0]; // update agent direction
  //os[0][LN.dY] = actionClone[1];
  os[0][LN.sX] = actionClone[2]; // update agent speed in next_state
  os[0][LN.sY] = actionClone[3];

  const mx = os[0][LN.aX];
  const my = os[0][LN.aY];
  //console.log(`mx: ${mx}, my: ${my}`);
  
 // if (currentStep == 0 && (parseInt(UI.episodeSlider.value) > Game.maxDrawEpisodes)) {
  //  Game.entities.splice(0, agent.batch_size);
 // }
  //if (Game.agentMoves.length > (agent.batch_size * Game.maxDrawBatches))

  //animateAgent();

  //console.log(`os2: ${observationSpace.next_stateSpace}`);
  //const reward = calculateReward();
  let isDone = false;
  let reward = 0;
  let penalty = 0;
  //let zomDist = utilsAI.distance(os[0][0],os[0][1],os[0][2],os[0][3]);
 // let civDist = utilsAI.distance(os[0][0],os[0][1],os[0][2],os[0][3]); 
  let civDist = utilsAI.distance(os[0][LN.aX], os[0][LN.aY], os[0][LN.cX], os[0][LN.cY]);
  let civAngle = utilsAI.angle([os[0][LN.aX], os[0][LN.aY]], [os[0][LN.cX], os[0][LN.cY]]);
  let agentHeading = utilsAI.angle([state[0][LN.aX], state[0][LN.aY]], [os[0][LN.aX],os[0][LN.aY]]);
  //let angleDegDif = Math.abs((civAngle * (180 / Math.PI)) - (agentHeading * (180 / Math.PI)));

  function angleDifferenceRadians(angle1, angle2) {
    let diff = Math.abs(angle1 - angle2);
    return Math.min(diff, 2 * Math.PI - diff);
}
let angleRadDif = angleDifferenceRadians(civAngle, agentHeading); // angleRadDif < ? or 0 + reward

  //let angleDeg1 = (civAngle + Math.PI) / (2 * Math.PI); // Normalize to [0, 1];
  //let angleDeg2 = (os[0][7] + Math.PI) / (2 * Math.PI);
  //let angle1 = civAngle * (180 / Math.PI);
  //let angle2 = agentHeading * (180 / Math.PI);
  //console.log(`civAngle: ${civAngle}`);
  //console.log(`agentHeading: ${agentHeading}`);
  //console.log(`angle diff: ${angleRadDif}`);
  //let angleDegDif = Math.abs(angleDeg1 - angleDeg2);
   // let angleDegDif = Math.abs(civAngle - os[0][5]);
  //let angleDeg1 = civAngle * (180 / Math.PI);
  //let angleDeg2 = os[0][5] * (180 / Math.PI);
  //let angleDegDif = Math.abs(angleDeg1 - angleDeg2);
  //let angleRadDiff = angleDegDif * (Math.PI / 180);
   //1rad × 180/π = 57.296°
  //1° × π/180 = 0.01745rad
  // Assuming angleDegDif is in the range [0, 180]
  //let angleReward = Math.abs(180 - angleRadDif); // Reward increases as angleDegDif approaches 0
  // Scale the angle reward if needed
  //let angleRewardScaling = 0.003; // Adjust as needed 0.005
 //angleReward *= angleRewardScaling;

  let anglePenalty = angleRadDif; // Reward increases as angleDegDif approaches 0
  // Scale the angle reward if needed
  let anglePenaltyScaling = 0.2; // Adjust as needed 0.005
  anglePenalty *= anglePenaltyScaling;
  
  //let angleReward = 0;
  // Penalty for significant changes in angle
  //let scaledAngleDegDif = angleDegDif * 100;
  //let angleChangePenalty = 2.0; // Adjust as needed
  //let angleStabilityReward = Math.exp(-angleChangePenalty * scaledAngleDegDif); // Reward increases as angleDegDif approaches 0
  //angleStabilityReward = angleStabilityReward.toFixed(12);
  /*
  if (angleDegDif > 1.5) {angleReward -= 0.5}
  else if (angleDegDif > 1.0) {angleReward -= 0.3}
  else if (angleDegDif > 0.5) {angleReward += 0.3}
  else {angleReward += 0.5}
  */
  //let distanceReward = 0;
  //let distanceReward = 1 / civDist; // this one is for rewards
  let distancePenalty = civDist; 
  // Scale the distance reward if needed
  let distancePenaltyScaling = 0.001; // Adjust as needed 0.1 // set as penalty?
  distancePenalty *= distancePenaltyScaling;
  //distanceBase *= 100;

  //const stateDifX = Math.abs(state[0][LN.aX] - os[0][LN.cX]); // agent x,y - civ x,y
  //const stateDifY = Math.abs(state[0][LN.aY] - os[0][LN.cY]);
  //const n_stateDifX = Math.abs(os[0][LN.aX] - os[0][LN.cX]);
  //const n_stateDifY = Math.abs(os[0][LN.aY] - os[0][LN.cY]);

  //if (n_stateDifX < stateDifX) {distanceReward += 0.4}
  //else if (n_stateDifX > stateDifX) {penalty += 0.5}
  //if (n_stateDifY < stateDifY) {distanceReward += 0.4}
  //else if (n_stateDifY > stateDifY) {penalty += 0.5}

  //if (civDist < os[0][LN.dist]) {distanceReward += 0.1}
  //if (civDist > os[0][LN.dist]) {penalty += 1}
  
  //let totalReward = distanceReward + distanceBase + angleReward;
  //let totalReward = distanceBase + angleReward;
  
  //reward += totalReward;
  //reward += distancePenalty;
  penalty += anglePenalty + distancePenalty;
  /*
  //const minVal = 0; // having negative here allows a possitive reward can scale into a negative
  //const maxVal = 3;

  function scaleToMinusOneToOne(x, minVal, maxVal) {
    let scaledValue = -1 + 2 * (x - minVal) / (maxVal - minVal);
    return Math.min(1, Math.max(-1, scaledValue)); // if values go over they should clamp to -1 or 1
}
*/
function scaleZeroOne(x) {
  return x / (x + 1);
}
function scaleWithTanh(x) {
  return Math.tanh(x);
}
function calculateTimePenalty(step, maxSteps) {
  const maxPenalty = -0.1; // Adjust as needed
  const timeRatio = step / maxSteps;
  const timePenalty = Math.max(maxPenalty, maxPenalty * timeRatio);
  return timePenalty;
}
  //reward -= angleDegDif;
  //console.log(`step: ${n_steps}, civ Dist: ${civDist}`);
  //console.log(`distanceBase: ${distanceBase}`);
  //console.log(`Distance Penalty: ${distancePenalty}`);
  //console.log(`civAngle: ${civAngle}`);
  //console.log(`angleDeg 1: ${angleDeg1}`);
  //console.log(`angleDeg 2: ${angleDeg2}`);
  //console.log(`angleDegDif: ${angleDegDif}`);
  //console.log(`scaled angle dif: ${scaledAngleDegDif}`);
  //console.log(`Angle Penalty: ${anglePenalty}`);
  //console.log(`stability Reward: ${angleStabilityReward}`);
  //console.log(`total Reward: ${totalReward}`);
  //console.log(`angleRadDiff: ${angleRadDiff}`);
  //if((civAngle - state[0][3]) > 0.2 ) {reward += 0.5}
  //else {reward += 0.5}
  os[0][LN.dist] = civDist;
  os[0][LN.angle] = civAngle;
  //if (civDist <= (player.width * 10)) {reward += 0.1;}
  //if (civDist <= (player.width * 7)) {reward += 0.2;}
  //if (civDist <= (player.width * 4)) {reward += 0.3;}
  //if (civDist <= (player.width * 5)) {reward += 1;}
  if (civDist <= player.width) {
    reward += 50; console.warn("AGENT FOUND CIVILIAN!");
    Game.agentWins++;
    UI.agentWins.innerHTML = `Times Won: ${Game.agentWins}`;
    isDone = true;
  }
  if (hitWall) {penalty += 2}
  //const scaledReward = scaleZeroOne(reward);
  //const scaledPenalty = scaleZeroOne(penalty);
  //reward = scaledReward - scaledPenalty;

  //const maxSteps = agent.maxStepCount; 
  const timePenalty = calculateTimePenalty(currentStep, agent.maxStepCount);
 // if (stepNumber % 3 === 0) {}

  penalty += Math.abs(timePenalty);
  //console.log(`penalty: ${penalty}`);
  reward = reward - penalty;
  //console.log(`pre-scaled reward: ${reward}`);
  //reward = scaleWithTanh(reward);
  
  
  //reward = scaleToMinusOneToOne(reward, minVal, maxVal);
  //console.log(`scaled reward: ${scaledReward}`);
  //console.log(`time penalty: ${timePenalty}`);
  //console.log(`final penalty: ${penalty}`);
  //console.log(`final reward: ${reward}`);
 
  // Assuming angle is normalized between -1 and 1
//let normalizedAngle = civAngle / maxPossibleAngleValue; 

// Ensure that the angle reward is in a reasonable range
//normalizedAngle = Math.max(Math.min(normalizedAngle, 1), -1);

// Scaling factor for the angle reward
//let angleScaling = 0.1; // Adjust as needed

// Update the reward
//reward -= angleScaling * normalizedAngle;
  
  /*
  //let collider = collideCheck(actionClone,true);
  if (collider) {
    switch(collider){
      case "civilian": reward += 5; isDone = true;
      break;
      case "zombie": reward -= 2; isDone = true;
      break;    
    }
  }
  */

  
  
  const base_Next_State = JSON.parse(JSON.stringify(observationSpace.next_stateSpace));
  const next_State = normalizeData(base_Next_State);
  //console.log(next_State);
  // Return the next state, reward, and whether the game is done
  //const nextState = getGameState(); // Implement this to return the game state
  if (currentStep >= agent.maxStepCount) {isDone = true}
  let terminalFlag = false;
  if (isDone) {terminalFlag = true}
  Game.agentMoves.push([mx, my, terminalFlag]); // Fore drawing agent paths
  return { next_state: next_State, reward: reward, isDone: isDone };
}

// Math.seedrandom() ?

const agent = new Agent(actionSpace.numberActions, observationSpace.stateSpace[0].length, observationSpace.stateSpace[0].length + actionSpace.numberActions); 
let episodes = 5; ///100 - 2000 // check batches size
//agent.batch_size = 32; // shortcut. Batch size for memory buffer. these are overidden below.
//agent.maxStepCount = 32; // shortcut. steps to take per episode.
const epReward = [];
const totalAvgReward = [];
//let target = false;

function main(epNum,stepSize,batchSize,warmupSteps) {  // Removed async   // 
  let target = false;
  if (epNum && !isNaN(epNum) && epNum > 0) {episodes = epNum}
  if (stepSize && !isNaN(stepSize) && stepSize > 0) {agent.maxStepCount = stepSize}
  if (batchSize && !isNaN(batchSize) && batchSize > 0) {agent.batch_size = batchSize}
  if (warmupSteps && !isNaN(warmupSteps) && warmupSteps > 0) {agent.warmup = warmupSteps}
  //if(agent.memory.cnt == 0) {observationSpace.initUpdate();}
  observationSpace.initUpdate()
  for (let s = 0; s < episodes; s++) {
    if (!Game.running) {break}
    if (target) {
      break;
    }
    // STEP 0: reset everything
    let totalReward = 0;
    let state = envReset(); // make sure is right shape!
    let done = false;
    let currentStep = 0;
    let stepDone = false;
   
    while (!done) {
      if (!Game.running) {break}
      //console.log(currentStep);
      //console.log(state);
      //console.log(`State ${n_steps}: ${state}, shape: ${}`);
      const normalState = normalizeData(state);
      //console.log(`normState ${n_steps}: ${normalState}`);
      //console.log(normalState);
      // STEP 1a: get an action based on the current state 
      const action = agent.act(normalState); // choose_action(observation), is envReset(). // video 2 adds more noise to the action
      //console.log(state[0][6]);
      //console.log(action); // not a tensor, just array. [0,0,0,0]
      //console.log(`first state: ${state}`);
      // STEP 2a: step the environment with the action, returning the new state, rewards, and if done
      const { next_state, reward, isDone } = envStep(action, state, currentStep);
      //console.log(`reward: ${reward}`)
      // remember to consider what happens with no training
      // STEP 3: save the new state to the memory buffer
      //console.log(`state after env: ${state}`);
      //console.log(`next_state after env: ${next_state}`);
      //console.log(`action after env: ${action}`);
      //console.log(`saving isDone as: ${isDone}`);
      //console.log(reward);
      const normal_Next_State = normalizeData(next_state);
      //const normalReward = normalizeData([reward]);

      agent.savexp(normalState, normal_Next_State, action, isDone, reward); 
      
      // Step 4-15..: train the system
      agent.train(); // Removed Await // only gets called if memory.cnt = agent.batch size

      // STEP 16: make the current state the new state
      state = JSON.parse(JSON.stringify(observationSpace.next_stateSpace)); // DEEP COPY
      
      totalReward += reward;
      //console.log(`Reward: ${totalReward}`);
      
      if (currentStep >= agent.maxStepCount) {stepDone = true} // forcing batch size episodes
      currentStep++

      if (isDone || stepDone) {
        epReward.push(JSON.parse(JSON.stringify(totalReward)));
        const avgReward = epReward.slice(-100).reduce((a, b) => a + b, 0) / Math.min(epReward.length, 100);
        totalAvgReward.push(avgReward);

        //if avg_score > best score : best_score = avg_score; agent.save_models()

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
