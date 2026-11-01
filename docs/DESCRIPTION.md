# Midterm Project: Go Big or Go Home!

By this point, you are likely sufficiently familiar with pretraining LLMs. Yet, how should we know how much we’ve improved? Some people would say you shouldn’t compare yourself to other people, but they are definitively wrong. We are *asian*, after all. This midterm project aims to put our urge to dominate leaderboards to the global scale. Let’s participate in speedrunning NanoGPT! 

Specifically, this project is to participate in track 3 of the Modded-NanoGPT speedrun competition. NanoGPT speedrun was a competition hosted by Andrej Karpathy, where participants compete to reach 3.28 validation loss on training GPT-2. This was later extended by Keller Jordan (Muon’s author!) into the Modded NanoGPT speedrun, which had more diverse competition tracks and increased available compute.

The track link can be found here: https://github.com/KellerJordan/modded-nanogpt/blob/master/records/track_3_optimization/README.md

Track 3 is the *optimization* track, where participants compete on the minimum *steps* required to train a 162M model based on a modified GPT-2 architecture. The competition fixes model architecture, dataset and batch size, and the rest is free for all! 

The original competition is designed around 8x H100s. Since we don’t have H100s available, you are to use 8x H200s instead. To match the original competition’s training environment, you may not utilize more than 80GB of VRAM in any point of your run. Any run that has W&B loggings that indicate this will be disqualified. Since H200s mainly differ from H100s in VRAM capacity and bandwidth, your results should transfer to H100s without much variance.

We will still host an internal leaderboard for this project. Unlike labs, you do not need to submit your final model to the OJ. Instead, you should log your training runs to the wandb project  `midterm-nanogpt` , and ensure each run has theese summary metrics: `total_train_steps` , which records the final training steps, and `val_loss` , which contains the final validation loss. The leaderboard will read directly from wandb.
