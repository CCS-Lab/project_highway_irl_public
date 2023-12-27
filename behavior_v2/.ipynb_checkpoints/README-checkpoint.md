# Direction to run the task
1. Copy **\_\_init\_\_.py** and **irl_env.py** to envs of highway_env lib. \
   (ex: if you install highway_env through anaconda, the **envs** path is **anaconda3/lib/python3.8/site-packages/highway_env/envs**)
2. Create a **task_code** folder in highway_env path (ex: **anaconda3/lib/python3.8/site-packages/highway_env/**). 
3. Copy **irl\_graphics.py**, **irl\_highway\_road.py**, and **irl\_vehicle.py** to the **task_code** folder.
4. Now, you can use the task environment by using **env = gym.make("IRL-v0")**. \
   Please refer to the example.ipynb.
