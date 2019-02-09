# CS7641_A1

### **Project 3**  

&nbsp;&nbsp;Replicate Figure 3a, 3b, 3c and 3d from Amy Greenwald/Keith Hall's 2003 "Correlated-Q" paper   
  
&nbsp;&nbsp;The code was written in Python version 3.6.7

&nbsp;&nbsp;Python packages dependency:  
&nbsp;&nbsp;**numpy 1.15.4**  
&nbsp;&nbsp;**pandas 0.23.4**  
&nbsp;&nbsp;**cvxopt 1.2.0**  
&nbsp;&nbsp;**matplotlib 3.0.1**  

&nbsp;&nbsp;usage: python p3.py [-h] [-l] learner  
&nbsp;&nbsp;learner: "Q", "friend-Q", "foe-Q", "uCE-Q" 

&nbsp;&nbsp;example:  
&nbsp;&nbsp;for Q-learner:  
&nbsp;&nbsp;python p3.py -l Q  


&nbsp;&nbsp;for uCE-Q:  
&nbsp;&nbsp;python p3.py -l uCE-Q


### **Project 2**  

&nbsp;&nbsp;Implement a Reinforcement Learning agent to solve the Lunar Lander openAI gym environment   
  
&nbsp;&nbsp;The code was written using Python version 3.7.0

&nbsp;&nbsp;Python packages dependency:  
&nbsp;&nbsp;**numpy 1.15.2**  
&nbsp;&nbsp;**pandas 0.23.4**  
&nbsp;&nbsp;**scipy 1.1.0**  
&nbsp;&nbsp;**gym 0.10.8**  
&nbsp;&nbsp;**torch 0.4.1**  

&nbsp;&nbsp;usage: lunarlander_ppo.py [-h] [--test] [--load LOAD]


&nbsp;&nbsp;optional arguments:
  -h,&nbsp;--help&nbsp;&nbsp;show this help message and exit  
  --test&nbsp;&nbsp;Do 10 random tests with a saved model otherwise train  
  --load&nbsp;LOAD&nbsp;&nbsp;path to saved model (.pth)  

&nbsp;&nbsp;example:  
&nbsp;&nbsp;for testing: python ppo_learn.py --test --load C:\pth\solved_1208.pth  
&nbsp;&nbsp;for training: python ppo_learn.py  

&nbsp;&nbsp;11 trained models are located in \project2\pth folder  


### **Project 1**  

&nbsp;&nbsp;This is an attempt to replicate the results (Figure 3, 4 and 5) of the "Random Walk" experiements described in  
&nbsp;&nbsp;section 3.2 in Richard Sutton's 1988 "Learning To Predict by the Methods of Temporal Differences" paper  
  
&nbsp;&nbsp;The code was written using Python version 2.7

&nbsp;&nbsp;Python packages used:  
&nbsp;&nbsp;**numpy 1.11.0**  
&nbsp;&nbsp;**pandas 0.20.3**  
&nbsp;&nbsp;**matplotlib 2.0.2**   

&nbsp;&nbsp;Command to run the code:  
&nbsp;&nbsp;**python p1.py**
