// tf.memory.numTensors() // reduce t o1 round to find where memory leaks are.
//tidy() -->
// const result = tf.scalar(121);
// res1 = tf.keep(result.sqrt());
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
     // let normalizedAngle = (angle + Math.PI) / (2 * Math.PI); // Normalize to [0, 1]
    //normalizedAngle = normalizedAngle * 2 - 1; 
    
  }
}

//observationSpace.stateSpace.length
const observationSpace ={ 
  //[agentX,agentY,zombieX,zombieY,civilianX,civilianY,dist to civ, angle to civ]
  initUpdate: function() { // [0,0],[0,0]
    observationSpace.defaults[0][0] = player.x + (player.width/2);
    observationSpace.defaults[0][1] = player.y + (player.height/2);
    observationSpace.civLoc[0] = civ1.x + (civ1.width/2);
    observationSpace.civLoc[1] = civ1.y + (civ1.height/2);
  },
  civLoc:[civ1.x + (civ1.width/2), civ1.y + (civ1.height/2)],
  defaults: [[player.x + (player.width/2), player.y + (player.height/2), 28, 1.56]],  // IF THESE (x,y) CHANGE, CHANGE IN CANVAS ALSO!
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
    this.model.add(tf.layers.dense({ inputShape: [inputShape], units: 400, activation: 'relu', dtype: 'float32', kernelInitializer: 'randomNormal'})); // state_dim + action_dim (video 2)
    this.model.add(tf.layers.dense({ units: 300, activation: 'relu', dtype: 'float32', kernelInitializer: 'randomNormal' }));
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
    this.model.add(tf.layers.dense({ inputShape: inputShape, units: 400, activation: 'relu', dtype: 'float32', kernelInitializer: 'randomNormal'  })); // input state_dim (video 2)
    this.model.add(tf.layers.dense({ units: 300, activation: 'relu', dtype: 'float32', kernelInitializer: 'randomNormal'   }));
    this.model.add(tf.layers.dense({ units: n_actions, activation: 'tanh', dtype: 'float32', kernelInitializer: 'randomNormal'  })); // output units: is number of actions/action space
    // from DDPG paper. pi - tangent hyperbolic, +-1
    // if action bounds are say +-2, multiply that by tanh function before predict/output
  }
  
  call(stateTensor) {
    //console.log(`stateTensor: ${stateTensor}`);
    // if action bounds not +-1 can multiply here
    //const normalState = normalizeData(stateTensor);
    const pred = this.model.predict(stateTensor);
    //console.log(`x: ${x}`);
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
    this.memory = new RBuffer(RBufferSize, [observationSpace.stateSpace[0].length], this.n_actions); // [stateSpace dimentions]
    //this.replace = 5;
    this.gamma = gamma;
    this.tau = tau;
    this.actor_update_steps = 2; 
    this.warmup = warmup; // initialy 200
    this.trainstep = 0;
    this.maxStepCount = 32; // not tied to trainstep
    this.min_action = actionSpace.low[0];   // negative movement
    this.max_action = actionSpace.high[0];  // positive movement

    //video sets noise here. and update_network_parameters(tau=1)
    this.critic_main.model.compile({optimizer: this.c_opt1, loss: tf.losses.meanSquaredError}); // , loss: tf.losses.meanSquaredError
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
  //console.log(`state tensor: ${stateTensor}, with shape: ${stateTensor.shape}`);
  // STEP 1b: get actions from actor_main
  let actions = this.actor_main.call(stateTensor);
  //let reshapedActions = tf.reshape(tf.clone(actions),[2]);
  //console.log(`Actor actions: ${actions}, with shape: ${actions.shape}`); 
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
      const ranNormalActions1 = tf.clone(tf.add(targetActions1, tf.clipByValue(tf.randomNormal(targetActions1.shape, 0.0, 0.2), -0.5, 0.5)));//.map(tf.clone); // video 2 uses actionsTensor?
      //console.log(`randomNormal actions: ${ranNormalActions1}, ${ranNormalActions1.shape}`);
      const clipActions1 = tf.clone(tf.mul(tf.scalar(agent.max_action), tf.clipByValue(ranNormalActions1, agent.min_action, agent.max_action)));//.map(tf.clone); 
      //console.log(`cliped actions: ${clipActions1}, ${clipActions1.shape}`);
      // STEP 7: The two Critic targets take each the couple (s`, a`) as input and return two Q-values as outputs
      const targetNextStateValues1_1 = agent.critic_target.call(nextStatesTensor, clipActions1).squeeze([1]);
      const targetNextStateValues1_2 = agent.critic_target2.call(nextStatesTensor, clipActions1).squeeze([1]);
      // Shape is [batch_size, 1], want to collaps to [batch_size]. (squeeze)
      // STEP 8: we keep the minimum of the two Q-values. min(Qt1, Qt2)
      const nextTargetStateValue1 = tf.minimum(targetNextStateValues1_1, targetNextStateValues1_2);
      //console.log(`NTSV: ${nextTargetStateValue1}, ${nextTargetStateValue1.shape}`);
      // STEP 9: we get the final target of the two Critic models (Qt = r + gamma * min(Qt1, Qt2) * dones, where gamma is the discount factor)
      //const targetValues1 = rewards1.add(nextTargetStateValue1.mul(tf.scalar(agent.gamma).mul(donesTensor))); //Phil does 1-dones,mines at save
      const targetValues1 = tf.add(rewards1, tf.mul(tf.scalar(agent.gamma), tf.mul(nextTargetStateValue1, donesTensor)));
      // Phil says "that will set the value of the second term, gamma*critic value to 0 everyhwere the done flag is true"
      //console.log(`targetValues: ${targetValues1}, ${targetValues1.shape}`);
      // says we have to squeeze because we have batch dimentino, and doesn't learn if you past that through.
      const criticValue1 = agent.critic_main.call(statesTensor, actionsTensor).squeeze([1]); // squeese? says wont learn otherwise
      //console.log(`criticValue: ${criticValue1}, ${criticValue1.shape}`);
      const criticLoss1 = tf.losses.meanSquaredError(criticValue1, targetValues1);
      //console.log(`crit Loss1: ${criticLoss1}, ${criticLoss1.shape}`);
     
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
  // Method for saving the model
  saveModel() {

  }
  // Method for loading a model
  loadModel() {

  }
} // End Agent Class
/*
function envReset() {
  // Initialize or reset the game environment and entities

 // numbersCopy = JSON.parse(JSON.stringify(nestedNumbers)); // DEEP COPY ARRAY CODE
  //observationSpace.resetTarget();
  observationSpace.stateSpace = JSON.parse(JSON.stringify(observationSpace.defaults));
  observationSpace.next_stateSpace = JSON.parse(JSON.stringify(observationSpace.defaults));
  const os = observationSpace.stateSpace;
  const inputMax = os.max();
  const inputMin = os.min();

  const normalizedData = os.sub(inputMin).div(inputMax.sub(inputMin));
  
  return normalizedData
  
}
*/

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
  // Initialize or reset the game environment and entities

  // Reset the state space
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
  //console.log(`os1: ${observationSpace.next_stateSpace}`);
  //console.log(os);
  //console.log(action);
   // Calculate new positions of entities, resolve collisions, etc.
   const x = (playerSpeed * actionClone[0]); //ex. (5 * 1 = 5, 5 * 0.5 = 2.5)
   const y = (playerSpeed * actionClone[1]); //.toFixed(5)
   //console.log(`x: ${x}, y: ${y}`);
   //console.log(`osX: ${os[0][0]}, osY: ${os[0][1]}`);
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
  //console.log(`mx: ${mx}, my: ${my}`);
  Game.agentMoves.push([mx, my])
  //animateAgent();

  //console.log(`os2: ${observationSpace.next_stateSpace}`);
  //const reward = calculateReward();
  let isDone = false;
  let reward = 0;
  
  //let zomDist = utilsAI.distance(os[0][0],os[0][1],os[0][2],os[0][3]);
 // let civDist = utilsAI.distance(os[0][0],os[0][1],os[0][2],os[0][3]); 
  let civDist = utilsAI.distance(os[0][0],os[0][1],observationSpace.civLoc[0],observationSpace.civLoc[1]);
  let civAngle = utilsAI.angle([os[0][0],os[0][1]],[observationSpace.civLoc[0],observationSpace.civLoc[1]]);
  let angleDeg1 = civAngle * (180 / Math.PI);
  let angleDeg2 = state[0][3] * (180 / Math.PI);
  let angleDegDif = Math.abs(angleDeg1 - angleDeg2);
  //let angleRadDiff = angleDegDif * (Math.PI / 180);

  // Assuming angleDegDif is in the range [0, 180]
let angleReward = 180 - angleDegDif; // Reward increases as angleDegDif approaches 0
// Scale the angle reward if needed
let angleRewardScaling = 0.005; // Adjust as needed
angleReward *= angleRewardScaling;

let distanceReward = 1 / civDist; // Reward increases as the agent gets closer
// Scale the distance reward if needed
let distanceRewardScaling = 0.1; // Adjust as needed
distanceReward *= distanceRewardScaling;
if (civDist > os[0][2]) {distanceReward -= 0.5}
else if (civDist < os[0][2]) {distanceReward += 0.5;}


let totalReward = distanceReward + angleReward;
// Update the agent's reward
reward += totalReward; 

  //reward -= angleDegDif;
  console.log(`step: ${n_steps}, civ Dist: ${civDist}`);
  console.log(`distanceReward: ${distanceReward}`);
  console.log(`civAngle: ${civAngle}`);
  console.log(`angleDegDif: ${angleDegDif}`);
  console.log(`angleReward: ${angleReward}`);
  console.log(`total Reward: ${totalReward}`);
  //console.log(`angleRadDiff: ${angleRadDiff}`);
  //if((civAngle - state[0][3]) > 0.2 ) {reward += 0.5}
  //else {reward += 0.5}
  os[0][2] = JSON.parse(JSON.stringify(civDist));
  os[0][3] = civAngle;
  if (civDist <= player.width) {reward += 5; console.warn("AGENT FOUND CIVILIAN!");isDone = true;}
  if (hitWall) {reward -= 1}
 
  console.log(`reward: ${reward}`);
  //1rad × 180/π = 57.296°
  //1° × π/180 = 0.01745rad
  // Assuming angle is normalized between -1 and 1
//let normalizedAngle = civAngle / maxPossibleAngleValue; 

// Ensure that the angle reward is in a reasonable range
//normalizedAngle = Math.max(Math.min(normalizedAngle, 1), -1);

// Scaling factor for the angle reward
//let angleScaling = 0.1; // Adjust as needed

// Update the reward
//reward -= angleScaling * normalizedAngle;



  //let lMX = (10 * Math.sin(angle));
  //let lMY = (10 * Math.cos(angle));
  //let moveX = (speed * Math.sin(angle));
  //let moveY = (speed * Math.cos(angle));
  //console.log(`civAngle: ${civAngle}`);
  //civLoc:[400,150],
  //console.log(os[0]);
  

  //if (civDist > os[0][2]) {reward -= 0.5}
 // else if (civDist < os[0][2]) {reward += 0.5;}
  
  //os[0][6] = observationSpace.targetDist
  //console.log(os[0][6]);

  
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
  if (n_steps >= agent.maxStepCount) {isDone = true}
  return { next_state: next_State, reward: reward, isDone: isDone };
}

// Math.seedrandom() ?

const agent = new Agent(actionSpace.numberActions, observationSpace.stateSpace[0].length, observationSpace.stateSpace[0].length + actionSpace.numberActions); 
const episodes = 5; ///100 - 2000 // check batches size
const epReward = [];
const totalAvgReward = [];
let target = false;

function main() {  // Removed async   // 
  observationSpace.initUpdate();
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
    let stepDone = false;
   
    while (!done) {
      if (!Game.running) {break}
      //console.log(n_steps);
      //console.log(state);
      //console.log(`State ${n_steps}: ${state}, shape: ${}`);
      const normalState = normalizeData(state);
      //console.log(`normState ${n_steps}: ${normalState}`);
      //console.log(normalState);
      // STEP 1a: get an action based on the current state 
      const action = agent.act(normalState); // choose_action(observation), is envReset(). // video 2 adds more noise to the action
      //console.log(state[0][6]);
      //console.log(action); // not a tensor, just array. [0,0] but why? for envStep?
      //console.log(`first state: ${state}`);
      // STEP 2a: step the environment with the action, returning the new state, rewards, and if done
      const { next_state, reward, isDone } = envStep(action, state, n_steps);
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
      
      if (n_steps >= agent.maxStepCount) {stepDone = true} // forcing batch size episodes
      n_steps++
      

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
