# TD3-TensorFlowJS
1) This is a passion project and is a work in progress, by a single person, and therefore may not work as expected. It is currently in development using the Google Chrome web browser, other browser might not work (I haven't tested any). You can check the "saved model data" folder for pre-trained models for the earlier builds, otherwise I am now including pre-trained models in the build folders when they are satisfactory.

2) I am creating this for the community, and therefore am hereby allowing anyone and everyone to use the code and information within for whatever (legal) purpose they desire (I'm using MIT License). I only ask that you give me / this repository (CloudZero2049) a shout-out if used in a project that is viewable to the community so that others can find it as well.
   
3) This is a list of mini-projects as part of the whole and their descriptions.
	3a) Find Target: The agent must find a civilian "target".  it is rewarded for reaching the civilian.
   
	3b) Run From Zombie: A more complex project. The agent must run from a zombie in a limited area, forcing it to run in large circles. It is rewarded based on how long it survives, and penalized if it 		touches the zombie.

5) The goal of this project is to create a working Twin Delayed Deep Deterministic Policy Gradient (TD3) (similar to DDPG), written in JavaScript with the only dependency being TensorFlow JS. It is meant to be a  blueprint that the community can use to create other projects with TensorFlow JS. The TD3 will use a simple environment containing an agent/player, and other entities such as a civilian or zombie. The environment is based on HTML5 Canvas. (x:0,y:0 in top left corner).
	4a) The agent has at least 2 actions [move direction x, move direction y].

	4b) The Observation Space has at least 2 array locations representing the x,y coordinates of the agent. It currently (and for the foreseeable future) has additional locations for "detection rays" that 	emit from the agent/player that show the distance to neerby object and entities (adjustable). Depending on the project, there might be more [[ax, ay, ray1, ray2, ray3, etc...]].

	4c) The projects use intricate reward systems to ensure that the agent is getting valid feedback for training. At the moment their are some rewards/penalties that are completely disabled if no sensor rays 	detect anything in order to prevent contamination.

	4d) Canvas animation is currently disabled when running TD3 (it would bog down training anyway). There is a visual representation that shows the paths the agent took after a run, and you can click the 	"Action Replay" button to see a full animation (after a run is finished). The 0 key will stop animation (good if there's looping errors tied to it).

 	4e) (Version 0.03) You can now record user inputs directly into the memory buffer to assist training if you wish.

   	4f) (Version 0.03) You can now train the agent directly from memory if you wish. This is much faster than going through the whole process and is usefull for when using user-generated memories (in 		moderation), otherwise it could have a negative effect on training because the agent isn't recording new actions.


6) This project began by using ChatGPT(v.3.5) to help me translate code from Python into JavaScript that utilizes TensorFlow JS. I could not find any examples of TD3 on the internet in JavaScript, this is the first for all I know..

7) This is the list of resources used during/in this project: 
	6a) TensorFlow.js API : https://js.tensorflow.org/api/latest/?_gl=1*191k82w*_ga*MTA3NDE5MTk2Ni4xNjk3NTIwNzU0*_ga_W0YLR4190T*MTY5OTE0MjIxNy4zMC4wLjE2OTkxNDIyMTcuMC4wLjA.

	6b) A link to The original TD3 paper: https://arxiv.org/pdf/1802.09477.pdf

	6c) Partial python TD3 code: https://towardsdatascience.com/deep-deterministic-and-twin-delayed-deep-deterministic-policy-gradient-with-tensorflow-2-x-43517b0e0185

	6d) full python TD3 code: https://github.com/abhisheksuran/Reinforcement_Learning/blob/master/td3withtau.ipynb

 	6e) OpenAI Gym Lunar Lander Documentation (for 5c, 5d environment): https://www.gymlibrary.dev/environments/box2d/lunar_lander/

   	6f) OpenAI Gym full Lunar Lander code on GitHub (for 5c, 5d environment): https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py

	6g) geeksforgeeks.org (helping learn TF.js functions): https://www.geeksforgeeks.org/

	6h) Machine learning with Phil, TD3 video: https://www.youtube.com/watch?v=ZhFO8EWADmY

	6i) Machine learning with Phil, TD3 TD3video2: https://www.youtube.com/watch?v=1lZOB2S17LU

	6j) Machine Learning with Phil, DDPG video: https://www.youtube.com/watch?v=4jh32CvwKYw

	6k) Machine Learning with Phil, everything to know about actor critic methods: https://www.youtube.com/watch?v=LawaN3BdI00

	6l) Justin Wallander, TD3 video: https://www.youtube.com/watch?v=-wq-luYhzy4

 	6m) Designing Rewards for Fast Learning, https://arxiv.org/pdf/2205.15400.pdf

   	6n) Reward (Mis)Design For Autonomous Driving, https://arxiv.org/pdf/2104.13906.pdf

8) About CloudZero (2023): I am a 32-year-old college student working towards an AA and thinking about a BA in Artifical Inteligence. My goal is to become a computer programmer of some sort. I have a deep interest in game design and artificial intelligence.
