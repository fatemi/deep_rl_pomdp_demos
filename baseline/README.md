# A Pixel-level Deep RL Implementation 

DQN and DDQN implementation for Catch environtment. 

The environment is borrowed from here[https://gist.github.com/EderSantana/c7222daa328f0e885093]. For more information, also see this blog[https://edersantana.github.io/articles/keras_rl/].

I made the envirnoment a bit cleaner, also added a `render()` method which makes it easier to use.

For the learning algorithm, there is a logical issue in the original code. Nevertheless, the original code works (because the issue behavious more or less similar to a reguralization term).

