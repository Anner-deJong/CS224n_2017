# CS224n 2017

ASSIGNMENT 4 NOT FINISHED YET

All homework assignments for the early 2017 Stanford **_CS224n: Natural Language Processing with Deep Learning_** class. <br />
**NB** Not including any data files

Great thanks to Chris Manning & Richard Socher (and the rest of the Stanford team) for publicly sharing the lectures, sheets and assignments for this class
([course website](http://web.stanford.edu/class/cs224n/index.html) & [lecture videos](https://www.youtube.com/playlist?list=PLqdrfNEc5QnuV9RwUAhoJcoQvu4Q46Lja)). <br />
I can highly recommend it for anybody aiming to learn more about this field.

This content is meant as references for your answers, as a means to debug, and sometimes for when it is unclear which code is right or wrong. There are a few more great answer decks out there, yet these are not consistent (nor is this one accordingly). I did however my best in cross checking my code and believe it is of decent standart.
I therefore hope anybody who is looking for references for their code will find this repository useful.

### Good luck and have fun!


p.s. **assignment 3 q3 (Grooving with GRUs) (b) ii.**
This repository does NOT include any answers to the written questions, as you can find those one the course's website itself.
One note however concerning the Grooving with GRUs assignment:

As far as my understanding goes, the answer provided on the website is wrong. <br />
A simple check, if x=1 & h_t-1=0, h_t should be 1. A U_z=1 (the official answer) 'activates' z, making the final h_t equal to h_t-1, equalling 0, which is wrong. <br />
Here are my personal answers, and I furthermore agree with the example given for this specific subquestion at [kedartatwawadi's GitBook](https://www.gitbook.com/book/kedartatwawadi/cs224n-assignment3/details).

0 < W_z <= |U_z| <br />
    U_z <= 0 <br />
0 < b_r <br />
    W_h <= -|U_h| <br />
0 < U_h <br />

p.p.s. I added a file q4_lstm_cell.py, which is not part of the homework. <br />
This code seems not compatible with with q3 (a) untill (e), the latchin and toggling behaviour of cells.

I think maybe somewhere down the code when passing tf.nn.dynamic_rnn the lstm cell, the state and output (the c and h) don't get passed properly. Hence I think the lstm png files are incorrect.

However, it seems to play well with the regular training, as (inside the results folder) the final F1 score on the full training is higher than for the gru cell (also 25% longer training time).
