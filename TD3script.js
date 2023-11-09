// tf.memory.numTensors() // reduce t o1 round to find where memory leaks are.
//tidy() -->
// const result = tf.scalar(121);
// res1 = tf.keep(result.sqrt());
const utilsAI = {
  distance: function(x1,y1,x2,y2) {
    return Math.floor(Math.hypot(x2 - x1, y2 - y1));
  },
}

//observationSpace.stateSpace.length
const observationSpace ={ 
  //resetTarget: function() {observationSpace.targetDist =  utilsAI.distance(observationSpace.defaults[0][0],observationSpace.defaults[0][1],observationSpace.defaults[0][4],observationSpace.defaults[0][5]);},
  //targetDist: 28, // adjusts from function
  //[agentX,agentY,zombieX,zombieY,civilianX,civilianY,dist to civ]
  defaults: [[50,50,30,30,70,70,28]],  // IF THESE (x,y) CHANGE, CHANGE IN CANVAS ALSO!
  stateSpace: [[50,50,30,30,70,70,28]],
  next_stateSpace: [[50,50,30,30,70,70,28]],
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
    this.done_memory = tf.zeros([maxsize]); // "terminal_memory of "done" flags"
  }
  
  storexp(state, next_state, action, done, reward) { // self, state, action, reward, state_, done
    const index = this.cnt % this.maxsize;
    done = done ? 1 : 0; // force numbers

    this.state_memory[index] = JSON.parse(JSON.stringify(state));
    this.next_state_memory[index] = JSON.parse(JSON.stringify(next_state));
    this.action_memory[index] = JSON.parse(JSON.stringify([action]));
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
    this.model.add(tf.layers.dense({ inputShape: [9], units: 400, activation: 'relu', kernelInitializer: 'randomNormal'})); // state_dim + action_dim (video 2)
    this.model.add(tf.layers.dense({ units: 300, activation: 'relu', kernelInitializer: 'randomNormal' }));
    this.model.add(tf.layers.dense({ units: 1, activation: null, kernelInitializer: 'randomNormal' })); // Output to 1 Q value
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
   
   //console.log(`catTensor: ${catTensor.shape}`);
    const x = this.model.predict(catTensor);
    
    return x;
  
  }
}
//kernelInitializer: tf.randomNormal(shape)
class Actor {
  constructor(n_actions,inputShape=[7]) { // has alpha in constructor (video)
    this.model = tf.sequential();
    this.model.add(tf.layers.dense({ inputShape: inputShape, units: 400, activation: 'relu', dtype: 'float32' })); // input state_dim (video 2)
    this.model.add(tf.layers.dense({ units: 300, activation: 'relu', dtype: 'float32'  }));
    this.model.add(tf.layers.dense({ units: n_actions, activation: 'tanh', dtype: 'float32' })); // output units: is number of actions/action space
    // from DDPG paper. pi - tangent hyperbolic, +-1
    // if action bounds are say +-2, multiply that by tanh function before predict/output
  }
  
  call(stateTensor) {
    //console.log(`ST: ${stateTensor}`);
    // if action bounds not +-1 can multiply here
    const x = this.model.predict(stateTensor);
    return x;
   
  }
}
//alpha,beta,input_dims,tau,env,gamma,update_actor_interval = 2, warmup = 1000, n_actions=2,max_size=1000000,layer1_size=400,layer2_size=300, batch_size=100,noise=0.1 (video)
//min-max actions is because of noise (video)

// alpha = learning rate for actor (.001), beta = learning rate for critic (.002), tau = weight update rate?
//gamma = discount factor (0 is immediate rewards, 1 is long term rewards and possibly more exploration)
//gamma = 0.99, alpha = 0.001
class Agent {
  constructor(n_actions = 2, cInputShape = 9, alpha = 0.001, beta = 0.002, gamma = 0.99, tau = 0.005, warmup = 32, RBufferSize = 100000) {                           
    this.actor_main = new Actor(n_actions);
    this.actor_target = new Actor(n_actions);
    this.critic_main = new Critic(cInputShape);
    this.critic_main2 = new Critic(cInputShape);
    this.critic_target = new Critic(cInputShape);
    this.critic_target2 = new Critic(cInputShape);
    this.batch_size = 32;
    this.n_actions = n_actions;
    this.a_opt = tf.train.adam(alpha);
    this.c_opt1 = tf.train.adam(beta);
    this.c_opt2 = tf.train.adam(beta);
    this.memory = new RBuffer(RBufferSize, [observationSpace.stateSpace[0].length], this.n_actions); // [stateSpace dimentions]
    //this.replace = 5;
    this.gamma = gamma;
    this.tau = tau;
    this.actor_update_steps = 2; 
    this.warmup = warmup; // initialy 200
    this.trainstep = 0;
    this.min_action = actionSpace.low[0];   // negative movement
    this.max_action = actionSpace.high[0];  // positive movement

    //video sets noise here. and update_network_parameters(tau=1)
    this.critic_main.model.compile({optimizer: this.c_opt1, loss: tf.losses.meanSquaredError});
    this.critic_main2.model.compile({optimizer: this.c_opt2, loss: tf.losses.meanSquaredError}); 
    this.actor_main.model.compile({optimizer: this.a_opt, loss: tf.losses.meanSquaredError}); 
    this.critic_target.model.compile({optimizer: this.c_opt1, loss: tf.losses.meanSquaredError});
    this.critic_target2.model.compile({optimizer: this.c_opt2, loss: tf.losses.meanSquaredError}); 
    this.actor_target.model.compile({optimizer: this.a_opt, loss: tf.losses.meanSquaredError}); 
    
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
  //const minTensor = tf.tensor(this.min_action);
  //const maxTensor = tf.tensor(this.max_action);
  //let shape = stateTensor.shape;
  //console.log(`first state tensor: ${stateTensor}, with shape: ${shape}`);
  // STEP 1b: get actions from actor_main
  const actions = this.actor_main.call(stateTensor);
  //let aShape = actions.shape;
  //console.log(`Actor actions: ${actions}`); // , with shape: ${aShape}
  if (!evaluate) { // meaning it is training
    const noise = tf.randomNormal([this.n_actions], 0.0, 0.1); // video uses mu, adds noise
    
    actions.add(noise);
  }
  //console.log(`act actions: ${actions}`);
  // video adds noise in here. clamping actions prevent noise causing going over or under
  //video sets state, mu, and my_prime
  // we clip actions because noise can cause them to go outside the bounds
  const clippedActions = tf.clipByValue(actions, this.min_action, this.max_action); 
  //const scaledActions = tf.mul(clippedActions, maxTensor); // mac_action isn't tensor, but is 1 anyway
  const scaledActions = clippedActions.mul(tf.scalar(this.max_action));
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
    let weights1 = [];
    //const targets1 = JSON.parse(JSON.stringify(this.actor_target.model.getWeights()));
    //const targets1 = this.actor_target.model.getWeights();
    let targets1 = this.actor_target.model.getWeights().map(tf.clone);
    
    //console.log(`target weights: ${targets1}`);
    let mainWeights1 = this.actor_main.model.getWeights().map(tf.clone);
    
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
    const rewardsTensor = tf.tensor(rewards); // no squeze?
    const actionsTensor = tf.tensor(actions).squeeze(); //actions needs [] here for some reason
    //console.log(`statesTensor:${statesTensor}, nextStatesTensor:${nextStatesTensor}, rewardsTensor${rewardsTensor}, actionsTensor:${actionsTensor}`);
    //console.log(`statesTensor:${statesTensor.shape}, nextStatesTensor:${nextStatesTensor.shape}, rewardsTensor${rewardsTensor.shape}, actionsTensor:${actionsTensor.shape}`);
    //console.log(`nextStatesTensor: ${nextStatesTensor}, shape: ${nextStatesTensor.shape}`);

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
      //console.log(`actor_target actions: ${targetActions}`);
      // STEP 6: We add Gaussian noise to the next action a` and we clamp it in a range of values supported by environment
      targetActions.add(tf.clipByValue(tf.randomNormal(targetActions.shape, 0.0, 0.2), -0.5, 0.5)); // video 2 uses actionsTensor?
      targetActions =  tf.clipByValue(targetActions, agent.min_action, agent.max_action).mul(tf.scalar(agent.max_action)); 
      //console.log(`clip actions: ${targetActions}`);
      // STEP 7: The two Critic targets take each the couple (s`, a`) as input and return two Q-values as outputs
      const targetNextStateValues = agent.critic_target.call(nextStatesTensor, targetActions).squeeze([1]);
      const targetNextStateValues2 = agent.critic_target2.call(nextStatesTensor, targetActions).squeeze([1]);
      // Shape is [batch_size, 1], want to collaps to [batch_size]. (squeeze)
      // STEP 8: we keep the minimum of the two Q-values. min(Qt1, Qt2)
      const nextTargetStateValue = tf.minimum(targetNextStateValues, targetNextStateValues2);
      
      //console.log(`NTSV: ${nextTargetStateValue}`);
      // STEP 9: we get the final target of the two Critic models (Qt = r + gamma * min(Qt1, Qt2) * dones, where gamma is the discount factor)
      const targetValues = rewardsTensor.add(nextTargetStateValue.mul(tf.scalar(agent.gamma).mul(dones))); //Phil does 1-dones,mines at save
      // Phil says "that will set the value of the second term, gamma*critic value to 0 everyhwere the done flag is true"
      //console.log(`targetValues: ${targetValues}`);
      // says we have to squeeze because we have batch dimentino, and doesn't learn if you past that through.
      const criticValue = agent.critic_main.call(statesTensor, actionsTensor).squeeze([1]) // squeese? says wont learn otherwise
      //console.log(`criticValue: ${criticValue}`);
      const criticLoss1 = tf.losses.meanSquaredError(targetValues, criticValue);
      //console.log(`crit Loss1: ${criticLoss1}`); // may be working properly
     
      return criticLoss1;
    };

    function lossFunction2() { 
       
       let targetActions = agent.actor_target.call(nextStatesTensor); 
       targetActions.add(tf.clipByValue(tf.randomNormal(targetActions.shape, 0.0, 0.2), -0.5, 0.5));
       targetActions = tf.clipByValue(targetActions, agent.min_action, agent.max_action).mul(tf.scalar(agent.max_action));
      
       const targetNextStateValues = agent.critic_target.call(nextStatesTensor, targetActions).squeeze([1]);
       const targetNextStateValues2 = agent.critic_target2.call(nextStatesTensor, targetActions).squeeze([1]);
       const nextTargetStateValue = tf.minimum(targetNextStateValues, targetNextStateValues2);
       const targetValues = rewardsTensor.add(nextTargetStateValue.mul(tf.scalar(agent.gamma).mul(dones)));

      const criticValue2 = agent.critic_main2.call(statesTensor, actionsTensor).squeeze([1]) // squeese? says wont learn otherwise?
      const criticLoss2 = tf.losses.meanSquaredError(targetValues, criticValue2);
    
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
  //observationSpace.resetTarget();
  observationSpace.stateSpace = JSON.parse(JSON.stringify(observationSpace.defaults));
  observationSpace.next_stateSpace = JSON.parse(JSON.stringify(observationSpace.defaults));
 
  return observationSpace.stateSpace
  
}

function envStep(action, n_steps) {
  const actionClone = JSON.parse(JSON.stringify(action));
  const playerSpeed = 5;  // temp hardcode 
  const os = observationSpace.next_stateSpace;
  //console.log(`os1: ${observationSpace.next_stateSpace}`);
  //console.log(os);
  //console.log(action);
   // Calculate new positions of entities, resolve collisions, etc.
  // move X
  if (actionClone[0] < 0) {os[0][0] -= playerSpeed}
  else if (actionClone[0] > 0) {os[0][0] += playerSpeed}
  // move Y
  if (actionClone[1] < 0) {os[0][1] -= playerSpeed}
  else if (actionClone[1] > 0) {os[0][1] += playerSpeed}
  //console.log(`os2: ${observationSpace.next_stateSpace}`);
  //const reward = calculateReward();
  let isDone = false;
  let reward = 0;
  
  //let zomDist = utilsAI.distance(os[0][0],os[0][1],os[0][2],os[0][3]);
  let civDist = utilsAI.distance(os[0][0],os[0][1],os[0][4],os[0][5]); 
  //console.log(os[0]);
  //console.log(`civ Dist: ${civDist}`);

  if (civDist > os[0][6]) {reward -= 2}
  else if (civDist < os[0][6]) {reward += 2;}
  os[0][6] = JSON.parse(JSON.stringify(civDist));
  //os[0][6] = observationSpace.targetDist
  //console.log(os[0][6]);
 
  let collider = collideCheck(actionClone,true); // this might not be working because dependencies
  if (collider) {
    switch(collider){
      case "civilian": reward += 5; isDone = true;
      break;
      case "zombie": reward -= 5; isDone = true;
      break;    
    }
  }

  //let reward = agent.trainstep;
  const next_State = JSON.parse(JSON.stringify(observationSpace.next_stateSpace));

  // Return the next state, reward, and whether the game is done
  //const nextState = getGameState(); // Implement this to return the game state
  if (n_steps >= agent.batch_size) {isDone = true}
  return { next_state: next_State, reward: reward, isDone: isDone };
}

// Math.seedrandom() ?

const agent = new Agent(actionSpace.numberActions,observationSpace.stateSpace[0].length); 
const episodes = 10; ///100 - 2000 // check batches size
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
      //console.log(n_steps);
      //console.log(`State ${n_steps}: ${state}`);
      // STEP 1a: get an action based on the current state 
      const action = agent.act(state); // choose_action(observation), is envReset(). // video 2 adds more noise to the action
      console.log(state[0][6]);
      console.log(action); // not a tensor, just array. [0,0] but why? for envStep?
      //console.log(`first state: ${state}`);
      // STEP 2a: step the environment with the action, returning the new state, rewards, and if done
      const { next_state, reward, isDone } = envStep(action, n_steps);
      // remember to consider what happens with no training
      // STEP 3: save the new state to the memory buffer
      //console.log(`state after env: ${state}`);
      //console.log(`next_state after env: ${next_state}`);
      //console.log(`action after env: ${action}`);
      //console.log(`saving isDone as: ${isDone}`);
      
      agent.savexp(state, next_state, action, isDone, reward); 
      
      // Step 4-15..: train the system
      agent.train(); // Removed Await // only gets called if memory.cnt = agent.batch size

      // STEP 16: make the current state the new state
      state = JSON.parse(JSON.stringify(observationSpace.next_stateSpace)); // DEEP COPY
      
      totalReward += reward;
      //console.log(`Reward: ${totalReward}`);
      if (n_steps >= agent.batch_size) {batchSteps = true} // forcing batch size episodes
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
