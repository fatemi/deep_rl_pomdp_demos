# A Pixel-level Deep RL Implementation 

DQN and DDQN implementation for Catch environtment. 

The environment is borrowed from [here](https://gist.github.com/EderSantana/c7222daa328f0e885093). For more information, also see this [blog](https://edersantana.github.io/articles/keras_rl/).

I made the envirnoment a bit cleaner, also added a `render()` method, which hopefully makes it an easier and general-purpose env to be used for both debugging and evaluation.

It would be quite usful for beginners to go through the blog post and the code. However, it should also be noted that for the learning algorithm, there is a logical issue/bug in the [original code](https://gist.github.com/EderSantana/c7222daa328f0e885093). (A zero target is used for other actions than the one selected. Nevertheless, it works because the issue behavious more or less similar to a reguralization term). Additionally, it does not have a target network. 

The code I release here, should be also useful from a more object-oriented viewpoint. 
