# TD3-TensorFlowJS
1) This is a passion project and is a work in progress, by a single person, and therefore may not work as expected. Also, expect a lot of comment code.

2) I am creating this for the community, and therefore am hereby allowing anyone and everyone to use the code and information within for whatever (legal) purpose they desire (I'm using MIT License). I only ask that you give me / this repository (CloudZero2049) a shout-out if used in a project that is viewable to the community so that others can find it as well.

3) The goal of this project is to create a working Twin Delayed Deep Deterministic Policy Gradient (TD3) (similar to DDPG), written in JavaScript with the only dependency being TensorFlow JS. It is meant to be a simple blueprint that the community can use to create other projects with TensorFlow JS. The TD3 will use a simple environment containing an agent/player, a zombie, and a civilian. The environment is based on HTML5 Canvas. (x:0,y:0 in top left corner)
	2a) The agent has 2 actions (move on the x coordinate, and y coordinate).

	2b) The Observation Space has 6 array locations representing the x,y coordinates of the agent, zombie, and civilian. [ax,ay,zx,zy,cx,cy]

	2c) Agent gets a positive reward for moving towards (and touching) the civilian, and a negative reward for moving away from civilian (and also for touching the zombie).

	2d) Canvas animation will remain disabled when running TD3 until it is stable and confirmed to be working properly, and without memory leaks. A "start player" button has been included that will animate while a human moves the player with WASD keys. The 0 key will stop animation (good for looping errors), and the 9 key will start it.


4) This project began by using ChatGPT(v.3.5) to help me translate code from Python into JavaScript that utilizes TensorFlow JS. I could not find any examples of TD3 on the internet in JavaScript, this is the first for all I know. I am not a professional with Python or TensorFlow at the time of writing.

5) This is the list of resources used during/in this project: 
	5a) TensorFlow.js API : https://js.tensorflow.org/api/latest/?_gl=1*191k82w*_ga*MTA3NDE5MTk2Ni4xNjk3NTIwNzU0*_ga_W0YLR4190T*MTY5OTE0MjIxNy4zMC4wLjE2OTkxNDIyMTcuMC4wLjA.

	5b) A link to The original TD3 paper: https://arxiv.org/pdf/1802.09477.pdf

	5c) Partial python TD3 code: https://towardsdatascience.com/deep-deterministic-and-twin-delayed-deep-deterministic-policy-gradient-with-tensorflow-2-x-43517b0e0185

	5d) full python TD3 code: https://github.com/abhisheksuran/Reinforcement_Learning/blob/master/td3withtau.ipynb

	5e) geeksforgeeks.org (helping learn TF.js functions): https://www.geeksforgeeks.org/

	5f) Machine learning with Phil, TD3 video: https://www.youtube.com/watch?v=ZhFO8EWADmY&list=WL&index=34&t=2364s

	5g) Machine learning with Phil, TD3 TD3video2: https://www.youtube.com/watch?v=1lZOB2S17LU&list=WL&index=33

	5h) machine Learning with Phil, DDPG video: https://www.youtube.com/watch?v=4jh32CvwKYw&list=PLuEC_Hx7ZKHXwwRVhrVVIqRH8DIoCgL39&index=30

	4j) Justin Wallander, TD3 video: https://www.youtube.com/watch?v=-wq-luYhzy4&list=PLuEC_Hx7ZKHXwwRVhrVVIqRH8DIoCgL39&index=29

6) About the owner (2023): I am a 32-year-old college student working towards an AA. My goal is to become a computer programmer of some sort. I have a deep interest in game design and artificial intelligence.
