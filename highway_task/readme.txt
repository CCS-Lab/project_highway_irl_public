1) installation

To install required dependencies, run the following commands at the task folder.
	
	pip install setuptools==62.6.0
	pip install e .

After the installation, executing "run_main.py" will start the task.

2) variables in run_main.py

duration_prac = length of the practice session (minutes)
duration = length of each block in the main session (minutes); There are two blocks in the task.

3) task settings in run_highway.py

You can change the task environment by modifying env.configure in run_highway.py
Below are the variables you may change

duration: episode duration in seconds
vehicles_count: number of other vehicles in an episode
vehicles_density: density of other vehicles; larger value = higher density
car_allocation: proportion of other cars on the three lanes
seed: random seed
initial_speed: speed of the own car at the start of an episode
max_speed: maximum speed of the own car

4) more description of the task
- You control the green car on the screen to drive as quickly as possible without crashing into yellow cars.
- Arrow keys are used to control the car; left = deceleration (speed -10), right = acceleration (speed +10), up = move to the adjacent upper lane, down = move to the adjacent lower lane. You can't control the car while changing lanes.
- Other cars start with 20 speed, and occasionally change their speed by 5 (max = 40). Cars on the same lane has the same speed. The minimum speed of the own car is 20.
- Once you overtake a car and make it unobservable on the screen, the overtaken car will not come back to the road.
- If you overtake every car within the duration of an episode, the episode ends with a bonus score. The next episode begins immediately without resetting the speed of the own car.
- The fuel on the top of the screen shows the remaining duration of the episode. If the fuel becomes zero, the own car stops and the next episode begins.