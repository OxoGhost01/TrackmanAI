# Hi everyone ! 

I'm trying here to create an AI witch can play tmnf (Trackmania Nation Forever), using Deep Reinforcement Learning.

## How it works 

To put it simply : you have a plugin running with the game (Python_Link.as). This plugin acts like a TCP server and connects to the client (game_instance_manager.py).
When on a race, the plugin send to the client some data about the car, and the agent i will implement later on, will decide what to do.

I use the TMI implementation from the [Linesight Project](https://github.com/Linesight-RL/linesight), because it's really hard to code and i don't want to spend too much time on this for now.
