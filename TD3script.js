// tf.memory.numTensors() // reduce t o1 round to find where memory leaks are.
//tidy() -->
// const result = tf.scalar(121);
// res1 = tf.keep(result.sqrt());
const utilsAI = {
  distance: function(x1,y1,x2,y2) {
    return Math.hypot(x1 - x2, y1 - y2);
  },
}

//observationSpace.stateSpace.length
const observationSpace ={ 
  resetTarget: function() {observationSpace.targetDist =  utilsAI.distance(observationSpace.defaults[0][0],observationSpace.defaults[0][1],observationSpace.defaults[0][4],observationSpace.defaults[0][5]);},
  targetDist: 500, // placeholder, adjusts from function
  //[agentX,agentY,zombieX,zombieY,civilianX,civilianY]
  defaults: [[50,50,30,30,70,70]],  // IF THESE CHANGE, CHANGE IN CANVAS ALSO!
  stateSpace: [[50,50,30,30,70,70]],
  next_stateSpace: [[50,50,30,30,70,70]],
}
//actionSpace.numberActions
//actionSpace.shape[1]
const actionSpace = {
  numberActions: 2,
  shape: [1,2], // not used
  actions: [0,0], // not used
  low: [-1,-1],
  high: [1,1]
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
    this.done_memory = tf.zeros([maxsize], 'bool'); // "terminal_memory of "done" flags"
  }

  storexp(state, next_state, action, done, reward) { // self, state, action, reward, state_, done
    const index = this.cnt % this.maxsize;
    
    this.state_memory[index] = JSON.parse(JSON.stringify(state));
    this.next_state_memory[index] = JSON.parse(JSON.stringify(next_state));
    this.action_memory[index] = JSON.parse(JSON.stringify(action));
    this.reward_memory[index] = JSON.parse(JSON.stringify(reward));
    this.done_memory[index] = JSON.parse(JSON.stringify(1 - Number(done))); // ? DDPG video is simply done. 1-2 = -1 ?
    this.cnt += 1;
  }

  sample(batch_size) {
    const max_mem = Math.min(this.cnt, this.maxsize);
    let batch;
    /*
    // setup an array the size of batch_size
    let tempBatches = Array.from(batch_size);
    // fill the array with random numbers (batch locations) up to max memory, duplicates possible
    for (let i = 0; i < tempBatches.length; i++) {
      tempBatches[i] = Math.floor(Math.random() * max_mem);
    }

    // Shuffle the array using the Fisher-Yates shuffle algorithm
    for (let i = tempBatches.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [tempBatches[i], tempBatches[j]] = [tempBatches[j], tempBatches[i]];
    }

    //const batch = Math.floor(Math.random() * (max_mem + 1)); // +1 was causes errors because array starts at 0
    // select a random batch from the final array
    batch = Math.floor(Math.random() * tempBatches.length);  // prevent double sampling? Erase bad memories?
    */
   //tf.util.shuffle (array)
    batch = Math.floor(Math.random() * max_mem);
    //console.log(`The batch: ${batch}, compare max_mem: ${max_mem}`);
    // tf.gather(x, indices, axis?, batchDims?) x = tensor
    const states = JSON.parse(JSON.stringify(this.state_memory[batch]));
    const next_states = JSON.parse(JSON.stringify(this.next_state_memory[batch]));
    const rewards = JSON.parse(JSON.stringify(this.reward_memory[batch]));
    const actions = JSON.parse(JSON.stringify(this.action_memory[batch]));
    const dones = JSON.parse(JSON.stringify(this.done_memory[batch]));
    //console.log(`chosen memory state; ${states}`);
    return [states, next_states, rewards, actions, dones];
  }
}

class Critic {  // it is possible to combine the 4 critics into 2 critics supposedly
  constructor(inputShape) { // completely different in video. checkpoints models to files here
    this.model = tf.sequential();
    this.model.add(tf.layers.dense({ inputShape: [8], units: 512, activation: 'relu' })); // state_dim + action_dim (video 2)
    this.model.add(tf.layers.dense({ units: 512, activation: 'relu' }));
    this.model.add(tf.layers.dense({ units: 1, activation: null })); // Output to 1 Q value
  }

  call(inputstate, action) { // feed forward
    //console.log(`inputstate: ${inputstate}`);
    //console.log(`action: ${action}`);
   
   // STEP 10b: we concatinate the input state[6] with action
   // Repeat the observation state tensor for each example in the batch
  const batchSize = action.shape[0];
  const repeatedObservationStateTensor = inputstate.tile([batchSize, 1]);
  // Concatenate the two tensors along the last axis (axis 1)
    const catTensor = tf.concat([repeatedObservationStateTensor, action], 1);
    //let oldCat = inputstate.concat(action); doesn't work, says shape is different
   
    //console.log(`catTensor: ${catTensor}`);
    const x = this.model.predict(catTensor);
    
    return x;
  
  }
}

class Actor {
  constructor(n_actions,inputShape=[6]) { // has alpha in constructor (video)
    this.model = tf.sequential();
    this.model.add(tf.layers.dense({ inputShape: inputShape, units: 512, activation: 'relu' })); // input state_dim (video 2)
    this.model.add(tf.layers.dense({ units: 512, activation: 'relu' }));
    this.model.add(tf.layers.dense({ units: n_actions, activation: 'tanh' })); // output units: is number of actions/action space
    // from DDPG paper. pi - tangent hyperbolic, +-1
    // if action bounds are say +-2, multiply that by tanh function before predict/output
  }
  
  call(stateTensor) {

    // if action bounds not +-1 can multiply here
    const x = this.model.predict(stateTensor);
    
    return x;
   
  }
}
//alpha,beta,input_dims,tau,env,gamma,update_actor_interval = 2, warmup = 1000, n_actions=2,max_size=1000000,layer1_size=400,layer2_size=300, batch_size=100,noise=0.1 (video)
//min-max actions is because of noise (video)

// alpha = learning rate for actor (.001), beta = learning rate for critic (.002), gamma = discount factor
class Agent {
  constructor(n_actions = 2, inputShape = 2, alpha = 0.001, beta = 0.002, gamma = 0.99, tau = 0.005, warmup = 150, RBufferSize = 100000) {                           
    this.actor_main = new Actor(n_actions);
    this.actor_target = new Actor(n_actions);
    this.critic_main = new Critic(inputShape);
    this.critic_main2 = new Critic(inputShape);
    this.critic_target = new Critic(inputShape);
    this.critic_target2 = new Critic(inputShape);
    this.batch_size = 32;
    this.n_actions = n_actions;
    this.a_opt = tf.train.adam(alpha);
    this.c_opt1 = tf.train.adam(beta);
    this.c_opt2 = tf.train.adam(beta);
    this.memory = new RBuffer(RBufferSize, [observationSpace.stateSpace.length], this.n_actions); // [stateSpace dimentions]
    //this.replace = 5;
    this.gamma = gamma;
    this.tau = tau;
    this.actor_update_steps = 2; 
    this.warmup = warmup; // initialy 200
    this.trainstep = 0;
    this.min_action = actionSpace.low;   // negative movement
    this.max_action = actionSpace.high;  // positive movement

    //video sets noise here. and update_network_parameters(tau=1)

    this.critic_target.model.compile({optimizer: this.c_opt1, loss: tf.losses.meanSquaredError}); // 
    this.critic_target2.model.compile({optimizer: this.c_opt2, loss: tf.losses.meanSquaredError}); // 
    this.actor_target.model.compile({optimizer: this.a_opt, loss: tf.losses.meanSquaredError}); // 
    
    //this.actor_target.model.compile({ optimizer: this.a_opt,loss: tf.losses.meanSquaredError,metrics: ['mse'], }); 
    //this.updateTarget(1); // tau = 1 for first update (from DDPG video)
  }

  act(state, evaluate = false) { // observation
  if (this.trainstep > this.warmup) {
    evaluate = true;
  }
  let returnValue = tf.tidy(() => {
  const stateTensor = tf.tensor(state); // [] added again because of DDPG video, then again because CGPT
  //let shape = stateTensor.shape;
  //console.log(`first state tensor: ${stateTensor}, with shape: ${shape}`);
  //const stateTensor = tf.tensor([state], [1,2], { dtype: 'float32' });
  // STEP 1b: get actions from actor_main
  const actions = this.actor_main.call(stateTensor);
  //let aShape = actions.shape;
  //console.log(`Actor actions: ${actions}`); // , with shape: ${aShape}

  if (!evaluate) { // meaning it is training
    const noise = tf.randomNormal([this.n_actions], 0.0, 0.1); // video uses mu, adds noise
    
    actions.add(noise);
  }
  // video adds noise in here. clamping actions prevent noise causing going over or under
  //video sets state, mu, and my_prime
  // we clip actions because noise can cause them to go outside the bounds
  const clippedActions = tf.clipByValue(actions, this.min_action, this.max_action); 
  const scaledActions = tf.mul(clippedActions, this.max_action); // mac_action isn't tensor, but is 1 anyway
 // console.log(scaledActions); // --> full tensor
 //console.log(scaledActions.arraySync()); //--> [4]
  // self.time_step += 1 (video)
  // Step 1c: returned action is the first element of a flatened clipped action array
  return scaledActions.arraySync()[0];  
}); // end tidy
return returnValue;
}

  savexp(state, next_state, action, done, reward) { // "remember" (video)
    this.memory.storexp(state, next_state, action, done, reward);
  }

  updateTarget(tau = null) { // numbersCopy = JSON.parse(JSON.stringify(nestedNumbers));
    if (tau == null) {
      tau = this.tau;
    }
    
    // STEP 14: we update the weights of the Actor target by polyak averaging
    let weights1 = [];
    //const targets1 = JSON.parse(JSON.stringify(this.actor_target.model.getWeights()));
    //const targets1 = this.actor_target.model.getWeights();
    let targets1 = this.actor_target.model.getWeights().map(tf.clone);
    //console.log(targets1);
    //console.log(`target weights: ${targets1}`);
    let mainWeights1 = this.actor_main.model.getWeights().map(tf.clone);
    //console.log(`main weights: ${mainWeights1}`);
    //console.log(`first weight: ${mainWeights1.getWeights()}`);
    for (let i = 0; i < mainWeights1.length; i++) {
      let updatedWeight = tf.tidy(() => {
        let weightedMain = mainWeights1[i].mul(tf.scalar(tau));
        //console.log(`weigthedMain ${weightedMain}`);
        let weightedTarget = targets1[i].mul(tf.scalar(1 - tau));
        //console.log(`weightedTarget ${weightedTarget}`);
        return weightedMain.add(weightedTarget);
      }); // end tidy
      //console.log(`updatedWeight ${updatedWeight}`);
      weights1.push(updatedWeight);
      
    } ;// end loop 1
    this.actor_target.model.setWeights(weights1);
    //console.log(`updated target model: ${this.actor_target.model.getWeights()}`);

    // STEP 15: we update the weights of the Critic targets by polyak averaging
    let weights2 = [];
    let targets2 = this.critic_target.model.getWeights().map(tf.clone);
    let mainWeights2 = this.critic_main.model.getWeights().map(tf.clone);
    for (let i = 0; i < mainWeights2.length; i++) {
      let updatedWeight = tf.tidy(() => {
        let weightedMain = mainWeights2[i].mul(tf.scalar(tau));
        let weightedTarget = targets2[i].mul(tf.scalar(1 - tau));
        return weightedMain.add(weightedTarget);
      }); // end tidy
      weights2.push(updatedWeight);
    } // end loop

    this.critic_target.model.setWeights(weights2);
   
    let weights3 = [];
    let targets3 = this.critic_target2.model.getWeights().map(tf.clone);
    let mainWeights3 = this.critic_main2.model.getWeights().map(tf.clone);
    for (let i = 0; i < mainWeights3.length; i++) {
      let updatedWeight = tf.tidy(() => {
        let weightedMain = mainWeights3[i].mul(tf.scalar(tau));
        let weightedTarget = targets3[i].mul(tf.scalar(1 - tau));
        return weightedMain.add(weightedTarget);
      }); // end tidy
      weights3.push(updatedWeight);
    } // end loop

    this.critic_target2.model.setWeights(weights3);
   

    //video does some weird dictionary stuff here
  
  }
  // STEPS 4-15...
  train() {
    
    if (this.memory.cnt < this.batch_size) {
        return;
    }
    tf.tidy(() => {
    // STEP 4: we sample a batch of transitions (s, s`, a, r) from memory
    const [states, nextStates, rewards, actions, dones] = this.memory.sample(this.batch_size);
    //console.log(`states:${states}, nextStates:${nextStates}, rewards:${rewards}, actions:${actions}, dones:${dones}`);
    
    // video has all these using critic_1, says doesnt matter. dtype matters
    const statesTensor = tf.tensor(states);// (CGPT)  const statesTensor = tf.tensor(states, [this.batch_size, ...this.statedim], 'float32');
    const nextStatesTensor = tf.tensor(nextStates);
    const rewardsTensor = tf.tensor(rewards);
    const actionsTensor = tf.tensor([actions]); //actions needs [] here for some reason
    //console.log(`statesTensor:${statesTensor}, nextStatesTensor:${nextStatesTensor}, rewardsTensor${rewardsTensor}, actionsTensor:${actionsTensor}`);
    //done = T.tensor(done).to(self.critic_1.device) // video has enabled?
    //dones = tf.convert_to_tensor(dones, dtype= tf.bool) 

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
      let targetActions = agent.actor_target.call(nextStatesTensor); 
     // console.log(`unmodified target actions from agent: ${targetActions}`);
      // STEP 6: We add Gaussian noise to the next action a` and we clamp it in a range of values supported by environment
      targetActions.add(tf.clipByValue(tf.randomNormal(targetActions.shape, 0.0, 0.2), -0.5, 0.5)); // video 2 uses actionsTensor?
      targetActions = tf.mul(agent.max_action, tf.clipByValue(targetActions, agent.min_action, agent.max_action)); 
      //console.log(`target actions: ${targetActions}`);
      // STEP 7: The two Critic targets take each the couple (s`, a`) as input and return two Q-values as outputs
      const targetNextStateValues = agent.critic_target.call(nextStatesTensor, targetActions).squeeze([1]);
      const targetNextStateValues2 = agent.critic_target2.call(nextStatesTensor, targetActions).squeeze([1]);
      // STEP 8: we keep the minimum of the two Q-values. min(Qt1, Qt2)
      const nextTargetStateValue = tf.minimum(targetNextStateValues, targetNextStateValues2);
      //console.log(`nextTargStateV: ${nextTargetStateValue}`);
      //console.log(`rewardsTensor: ${rewardsTensor}`);
      //console.log(`dones: ${dones}`);
      // STEP 9: we get the final target of the two Critic models (Qt = r + gamma * min(Qt1, Qt2) * dones, where gamma is the discount factor)
      const targetValues = rewardsTensor.add(nextTargetStateValue.mul(tf.scalar(agent.gamma).mul(tf.scalar(dones))));
      //console.log(`targetValues: ${targetValues}`);
      // says we have to squeeze because we have batch dimentino, and doesn't learn if you past that through.
      const criticValue = agent.critic_main.call(statesTensor, actionsTensor).squeeze([1]) // squeese? says wont learn otherwise
      //console.log(`criticValue: ${criticValue}`);
      const criticLoss1 = tf.losses.meanSquaredError(targetValues, criticValue);
      //console.log(`crit Loss1: ${criticLoss1}`); // may be working properly
      //const criticMean = criticLoss1.mean(); // criticLoss1 return full tensor, this gives mean of output
      //console.log(`TESTING loss: ${criticMean}`);
      return criticLoss1;
    };

    function lossFunction2() { 
       
       let targetActions = agent.actor_target.call(nextStatesTensor); 
       targetActions.add(tf.clipByValue(tf.randomNormal(targetActions.shape, 0.0, 0.2), -0.5, 0.5));
       targetActions = tf.mul(agent.max_action, tf.clipByValue(targetActions, agent.min_action, agent.max_action)); //.mul()??
      
       const targetNextStateValues = agent.critic_target.call(nextStatesTensor, targetActions).squeeze([1]);
       const targetNextStateValues2 = agent.critic_target2.call(nextStatesTensor, targetActions).squeeze([1]);
       const nextTargetStateValue = tf.minimum(targetNextStateValues, targetNextStateValues2);
       const targetValues = rewardsTensor.add(nextTargetStateValue.mul(tf.scalar(agent.gamma).mul(tf.scalar(dones))));

      const criticValue2 = agent.critic_main2.call(statesTensor, actionsTensor).squeeze([1]) // squeese? says wont learn otherwise?
      const criticLoss2 = tf.losses.meanSquaredError(targetValues, criticValue2);
      //const criticMean2 = criticLoss2.mean(); // criticLoss1 return full tensor, this gives mean of output. bad?
      //console.log(`TESTING loss2: ${criticMean2}`);
      return criticLoss2
      //return criticMean2;
    };

 
    // STEP 12: we backpropigate the Critic loss and update the parameters of the critic models through optiizers
    const gWeights1 = this.critic_main.model.getWeights(true)//.map(tf.clone);
    //console.log(`gWeights1: ${gWeights1}`);
    //this.c_opt1.computeGradients(lossFunction1,gWeights1);
    let computedgrads1 = this.c_opt1.computeGradients(lossFunction1,gWeights1);  // do we need to .step() optimizers?
    //console.log(computedgrads1);
    this.c_opt1.applyGradients(computedgrads1.grads);

    const gWeights2 = this.critic_main2.model.getWeights(true)//.map(tf.clone);
    let computedgrads2 = this.c_opt2.computeGradients(lossFunction2,gWeights2);
    this.c_opt2.applyGradients(computedgrads2.grads);
  
    this.trainstep += 1;

    // if self.learn_step_cntr % self.update_actor_iter != 0: return (is oposite)
    // STEP 13: once every two iterations, we update the Actor model by performing gradient ascent on the output of the first critic model
    if (this.trainstep % this.actor_update_steps === 0) {
      
      function lossFunction3() {
        // gradient ascent is the negative of gradient decent.
        const actorLoss = tf.mean(agent.critic_main.call(statesTensor, agent.actor_main.call(statesTensor)).neg());
        return actorLoss;
           // const new_policy_actions = self.actor_main(states)
           //  let actor_loss = -self.critic_main(states, new_policy_actions)
            // let actor_loss = tf.math.reduce_mean(actor_loss)
      }
      
      const gWeights3 = this.actor_main.model.getWeights(true)//.map(tf.clone);
      let computedgrads3 = this.a_opt.computeGradients(lossFunction3,gWeights3);
      //console.log(computedgrads3.grads);
      this.a_opt.applyGradients(computedgrads3.grads);
     
       // STEP 14/15... updating weights every two iterations. moved here because the original paper says to.
      this.updateTarget(); // same as self.update_netowrk_parameters() in video
      
    }
  }); // End Tidy

  
  }
  // Method for saving the model
  saveModel() {

  }
  // Method for loading a model
  loadModel() {

  }
} // End Agent Class

function envReset() {
  // Initialize or reset the game environment and entities

 // numbersCopy = JSON.parse(JSON.stringify(nestedNumbers)); // DEEP COPY ARRAY CODE
  observationSpace.resetTarget();
  observationSpace.stateSpace = JSON.parse(JSON.stringify(observationSpace.defaults));
  observationSpace.next_stateSpace = JSON.parse(JSON.stringify(observationSpace.defaults));
 
  return observationSpace.stateSpace
  
}

function envStep(action) {
  const playerSpeed = 5;  // temp hardcode 
  const os = observationSpace.next_stateSpace;

  //console.log(action);
   // Calculate new positions of entities, resolve collisions, etc.
  // move X
  if (action[0] < 0) {os[0][0] -= playerSpeed}
  else if (action[0] > 0) {os[0][0] += playerSpeed}
  // move Y
  if (action[1] < 0) {os[0][1] -= playerSpeed}
  else if (action[1] > 0) {os[0][1] += playerSpeed}

  //const reward = calculateReward();
  let isDone = false;
  let reward = 0;
  
  let collider = collideCheck(action,true); // this might not be working because dependencies
  if (collider) {
    switch(collider){
      case "civilian": reward += 100; isDone = true;
      break;
      case "zombie": reward -= 50; isDone = true;
      break;    
    }
  }

  //let zomDist = utilsAI.distance(os[0][0],os[0][1],os[0][2],os[0][3]);
  let civDist = utilsAI.distance(os[0][0],os[0][1],os[0][4],os[0][5]); 
  //console.log(observationSpace.targetDist);
  //console.log(`civ Dist: ${civDist}`);

  if (civDist >= observationSpace.targetDist) {reward -= 35}
  else {reward += 35;}
  observationSpace.targetDist = JSON.parse(JSON.stringify(civDist));
  //console.log(observationSpace.targetDist);
 
  //let reward = agent.trainstep;
  const next_State = JSON.parse(JSON.stringify(observationSpace.next_stateSpace));

  // Return the next state, reward, and whether the game is done
  //const nextState = getGameState(); // Implement this to return the game state
  return { next_state: next_State, reward: reward, done: isDone };
}

// Math.seedrandom() ?

const agent = new Agent(actionSpace.numberActions,observationSpace.stateSpace.length); 
  //console.log("At creation");
  //console.log(agent.c_opt1.getWeights());
const episodes = 50; ///100 - 2000 // check batches size
const epReward = [];
const totalAvgReward = [];
let target = false;

function main() {  // Removed async   // 
  for (let s = 0; s < episodes; s++) {
    if (!Game.running) {break}
    if (target) {
      break;
    }
    // STEP 0: reset everything
    let totalReward = 0;
    let state = envReset(); // make sure is right shape!
    let done = false;
    let n_steps = 0;
    let batchSteps = false;
   
    while (!done) {
      if (!Game.running) {break}
      // STEP 1a: get an action based on the current state 
      const action = agent.act(state); // choose_action(observation), is envReset(). // video 2 adds more noise to the action
      //console.log(`agent action: ${action}`); // not a tensor, just array. [0,0]
      //console.log(`first state: ${state}`);
      // STEP 2a: step the environment with the action, returning the new state, rewards, and if done
      const { next_state, reward, done: isDone } = envStep(action);
      // remember to consider what happens with no training
      // STEP 3: save the new state to the memory buffer
      //console.log(`state after env: ${state}`);
      //console.log(`next_state after env: ${next_state}`);
      agent.savexp(state, next_state, action, isDone, reward); 
      
      // Step 4-15..: train the system
      agent.train(); // Removed Await // only gets called if memory.cnt = agent.batch size

      // STEP 16: make the current state the new state
      state = JSON.parse(JSON.stringify(observationSpace.next_stateSpace)); // DEEP COPY
      
      totalReward += reward;

      //console.log(`Reward: ${totalReward}`);
      if (n_steps >= agent.batch_size) {batchSteps = true}
      n_steps++
      

      if (isDone || batchSteps) {
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
}
