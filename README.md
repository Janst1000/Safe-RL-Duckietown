# Safe-RL-Duckietown

Safe-RL-Duckietown is a project that uses safe reinforcement learning to train Duckiebots to follow a lane while keeping the robots safe during training.

## Introduction

This project is an early stage implementation of safe-reinforcment learning for Duckiebots. The goal of the projects was to let the reinforcement learning agent learn lane-following without going into unsafe states. For this, all positions outside the lane are considered unsafe. This project was part of a bachelors thesis and if you want to learn more about the theory and get more in-depth information on how the project works you can take a look at the [thesis](./docs/Safe_Reinforcement_Learning_in_Duckietown.pdf). Additionally you can also find a presentation [here](./docs/Safe-RL-Duckietown.pdf).

### How does it work?

The reinforcement learning implementation is based on a Deep-Q-Networks algorithm that takes in the distance from the center of the lane and the angle to the center line of the lane. The reward that the robot receives is based on a mathematical equation that takes into account two factors: the distance from the center of the lane and the angle to the center line of the lane. The equation is $reward = 1 - 3*lane_d^{2} - lane\_\varphi^{2}$, where lane_d is the distance from the center of the lane and lane_φ is the angle to the center line of the lane. The robot is rewarded more for being closer to the center of the lane and for being parallel with the lane.

To get the data from the robot, the lane filter nodes from the [dt-core](https://github.com/duckietown/dt-core) repository are being used. The dt-core repository is a collection of software modules that provide basic functionality for Duckietown robots, including lane detection and filtering. By using the lane filter nodes, we can extract the necessary information about the robot's position and orientation relative to the lane, which is then used to calculate the reward.

To keep the robot safe during learning, a safety layer was implemented. The safety layer is a simple linear regression model that takes in the action (linear_v, angular_v) and outputs the new expected position of the robot. The model is trained by collecting data by driving the robot around the lane and recording the position of the robot and the action taken by the robot, which is then used to train the model. During execution of the agent, the safety layer is then used to predict the new position of the robot given an action. If the predicted position is outside the lane, the action is not executed and the robot is either reconsiders or tries to recover to a safe state. This way the robot is kept safe during learning.

## Setup

This project relies heavily of the duckietown shell and it's docker implementation. It is recommended to read through the instructions of the duckietown shell first to fully be able to use this projects features. The duckietown shell can be found [here](https://docs.duckietown.com/daffy/opmanual-duckiebot/setup/setup_laptop/index.html).

### Installation

This project assumes that your robot is already setup and calibrated.

You should clone this repository and then also get the submodules. To do this you can use the following commands:

```bash
git clone https://github.com/Janst1000/Safe-RL-Duckietown.git
cd Safe-RL-Duckietown
git submodule update --init --recursive
```

Next you should build the project. For this you should decide if you want to run the code on the robot itself or if you want to run it on your laptop. If you want to run it on your laptop you can use the following command:

```bash
dts devel build -f
```

If you want to build it on the robot you can use the following command:

```bash
dts devel build -f -H ![ROBOT_NAME]
```

### Collecting data and training the model
Once you have your robot ready you should start collecting some data for the safety layer. To do this you can use the `collect_data.py` script. This script will drive the robot around the lane and record the position of the robot and the action taken by the robot. It is recommended to run this on the robot itself to remove any network delay. If you are planning to run the agent on your computer itself, you should use the `-R` option instead of the `-H` option. To run the script on the robot you can use the following command:

```bash
dts devel run -H ![ROBOT_NAME] -L collect_data
```

If you want to run it on your computer instead you can use the following command:

```bash
dts devel run -R ![ROBOT_NAME] -L collect_data
```

After the script is started, the robot will drive around randomly so you should put it back in the lane if it goes outside. If you think the robot collected enough data, you can stop the script with `Ctrl+C`. This will stop the script but the docker container is still active. So now you should copy the data to your computer to train the model. To do this you can use the following command:

```bash
docker cp dts-run-safe-rl-duckietown:/tmp/data.txt data.txt
```

After this you can stop the docker container with the following command:

```bash
exit
```
Now it's time to train the model. For this you run the `train_model.py` script. This script will train the model and save it to the `model.npz` file. This requires `numpy` and `sklearn` to be installed. To install these you can use the following command:

```bash
pip3 install numpy sklearn
```

To then run the script you can use the following command:

```bash
python3 train_model.py
```

Now you are ready to run the agent.

### Running the agent

The performance of the agent is way better when running on the robot itself so it is recommended to run it on the robot. You can either run it on your computer with the `-R` option or on the robot with the `-H` option. Additionally you can decide if you want to run the baseline agent or the safe agent. To run the safe agent you can use the following command:

```bash
dts devel run -H ![ROBOT_NAME]
```

To run the baseline agent you can use the following command:

```bash
dts devel run -H ![ROBOT_NAME] -L no_safety
```

The robot should start moving after some time. Whenever the robot is ready to train it's model, it will stop in place and start training. This will take some time so be patient. After the training is done, the robot will start moving again. If you want to stop the agent you can use `Ctrl+C` to stop the docker container.

## Results

To find some more detailed results you can look at the thesis document [here](./docs/Safe_Reinforcement_Learning_in_Duckietown.pdf). But to have a more visual comparison of the two agents, you can look at the following videos.

Here is the baseline agent:

<video width="640" height="360" controls>
  <source src="./assets/no-safety.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

And here is the safe agent with the safety layer:

<video width="640" height="360" controls>
  <source src="./assets/safety.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

There clearly is an improvement in safety but it still isn't perfect. The agent still leaves the lane once in the test but in all the other times it recovers by itself. For more information you can look at the thesis document.

### Future improvements

- The safety range for the `lane_d` could be adjusted to be more strict. This would make the agent more safe but also more conservative. By doing this the robot would keep the lane markings in it's field of view for longer but would also execute the recovery actions more often.
- The safety layer could be improved by using a more complex model. The current model is a simple linear regression model but a more complex model could be used to improve the performance.
- The input data can also easily be adjusted to include more information. For this we could include additional sensor data.
- Finally we can also try adjusting the reward function to penalize certain behaviours more or less. This could be used to make the agent more safe or more aggressive.
