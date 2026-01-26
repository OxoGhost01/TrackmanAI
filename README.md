# Hi everyone ! 

I'm trying here to create an AI witch can play tmnf (Trackmania Nation Forever), using Deep Reinforcement Learning.

## How it works 

To put it simply : you have a plugin running with the game (Python_Link.as). This plugin acts like a TCP server and connects to the client (game_instance_manager.py).
When on a race, the plugin send to the client some data about the car, and the agent decide what to do.

> More in detail later, when the agent will be ready, i'll make a video to explain how it works.

I use the TMI implementation from the [Linesight Project](https://github.com/Linesight-RL/linesight), because it's really hard to code and i don't want to spend too much time on this for now.
 -> all the files in TMI are from Linesight, maybe modified because of updates, and i simplified some things to make the code more readable.

