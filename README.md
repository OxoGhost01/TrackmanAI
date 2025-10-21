# Hi everyone ! 

I'm trying here to create an AI witch can play tmnf (Trackmania Nation Forever), using Deep Reinforcement Learning.

## How it works 

To put it simply : you have a plugin running with the game (Link.as). This plugin acts like a TCP server and connects to the client (game.py).
When on a race, the plugin send to the client some data about the car, and the agent i will implement later on, will decide what to do.
